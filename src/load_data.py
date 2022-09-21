import itertools
import os
from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators, get_LFPs,
                                         get_trial_time, make_epochs_dataframe,
                                         make_neuron_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import (get_data_structure,
                                              reconstruct_time)
from loren_frank_data_processing.DIO import get_DIO, get_DIO_indicator
from loren_frank_data_processing.tetrodes import get_LFP_filename
from loren_frank_data_processing.well_traversal_classification import (
    score_inbound_outbound, segment_path)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from ripple_detection.core import gaussian_smooth, get_envelope
from scipy.io import loadmat
from scipy.signal import filtfilt, hilbert
from scipy.stats import zscore
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import (ANIMALS, LINEAR_EDGE_ORDER, LINEAR_EDGE_SPACING,
                            SAMPLING_FREQUENCY, WTRACK_EDGE_ORDER,
                            WTRACK_EDGE_SPACING)
from track_linearization import get_linearized_position
from track_linearization import make_track_graph as _make_track_graph

logger = getLogger(__name__)


ENVIRONMENTS = {'lineartrack': 'linearTrack',
                'lineartack': 'linearTrack',
                'wtrack': 'wTrack'}


def get_track_segments(epoch_key, animals):
    '''

    Parameters
    ----------
    epoch_key : tuple
    animals : dict of namedtuples

    Returns
    -------
    track_segments : ndarray, shape (n_segments, n_nodes, n_space)
    center_well_position : ndarray, shape (n_space,)

    '''
    environment = np.asarray(
        make_epochs_dataframe(animals).loc[epoch_key].environment)[0]

    coordinate_path = os.path.join(
        animals[epoch_key[0]].directory,
        f'{ENVIRONMENTS[environment]}_coordinates.mat')
    linearcoord = loadmat(coordinate_path)['coords'][0]
    track_segments = [np.stack(((arm[:-1, :, 0], arm[1:, :, 0])), axis=1)
                      for arm in linearcoord]
    track_segments = np.concatenate(track_segments)
    _, unique_ind = np.unique(track_segments, return_index=True, axis=0)
    return track_segments[np.sort(unique_ind)]


def get_well_locations(epoch_key, animals):
    '''Retrieves the 2D coordinates for each well.
    '''
    environment = np.asarray(
        make_epochs_dataframe(animals).loc[epoch_key].environment)[0]

    coordinate_path = os.path.join(
        animals[epoch_key[0]].directory,
        f'{ENVIRONMENTS[environment]}_coordinates.mat')
    linearcoord = loadmat(coordinate_path)['coords'][0]
    well_locations = []
    for arm in linearcoord:
        well_locations.append(arm[0, :, 0])
        well_locations.append(arm[-1, :, 0])
    well_locations = np.stack(well_locations)
    _, ind = np.unique(well_locations, axis=0, return_index=True)
    return well_locations[np.sort(ind), :]


def make_track_graph(epoch_key, animals, convert_to_pixels=False):
    '''

    Parameters
    ----------
    epoch_key : tuple, (animal, day, epoch)
    animals : dict of namedtuples

    Returns
    -------
    track_graph : networkx Graph

    '''
    track_segments = get_track_segments(epoch_key, animals)
    nodes = track_segments.copy().reshape((-1, 2))
    _, unique_ind = np.unique(nodes, return_index=True, axis=0)
    nodes = nodes[np.sort(unique_ind)]

    edges = np.zeros(track_segments.shape[:2], dtype=np.int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id

    return _make_track_graph(nodes, edges)


def get_labels(times, time):
    ripple_labels = pd.DataFrame(np.zeros_like(time, dtype=np.int), index=time,
                                 columns=['replay_number'])
    for replay_number, start_time, end_time in times.itertuples():
        ripple_labels.loc[start_time:end_time] = replay_number

    return ripple_labels


def estimate_ripple_band_power(lfps, sampling_frequency):
    m = Multitaper(lfps.values, sampling_frequency=sampling_frequency,
                   time_halfbandwidth_product=1,
                   time_window_duration=0.020,
                   time_window_step=0.020,
                   start_time=lfps.index[0].total_seconds())
    c = Connectivity.from_multitaper(m)
    closest_200Hz_freq_ind = np.argmin(np.abs(c.frequencies - 200))
    power = c.power()[..., closest_200Hz_freq_ind, :].squeeze() + np.spacing(1)
    n_samples = int(0.020 * sampling_frequency)
    index = lfps.index[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]
    power = pd.DataFrame(power, index=index)
    return power.reindex(lfps.index)


def get_ripple_consensus_trace(ripple_filtered_lfps, sampling_frequency):
    SMOOTHING_SIGMA = 0.004
    ripple_consensus_trace = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)
    ripple_consensus_trace[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null])
    ripple_consensus_trace = np.sum(ripple_consensus_trace ** 2, axis=1)
    ripple_consensus_trace[not_null] = gaussian_smooth(
        ripple_consensus_trace[not_null], SMOOTHING_SIGMA, sampling_frequency)
    return np.sqrt(ripple_consensus_trace)


def get_adhoc_ripple(epoch_key, tetrode_info, position_time,
                     position_to_linearize):
    LFP_SAMPLING_FREQUENCY = 1500

    # Get speed in terms of the LFP time
    time = get_trial_time(epoch_key, ANIMALS)
    position_df = get_position_info(
        epoch_key, skip_linearization=True)
    new_index = pd.Index(np.unique(np.concatenate(
        (position_df.index, time))), name='time')
    position_df = (position_df
                   .reindex(index=new_index)
                   .interpolate(method='linear')
                   .reindex(index=time)
                   )
    speed_feature = position_to_linearize[0].split('_')[0]
    speed = position_df[f'{speed_feature}_vel']

    # Load LFPs
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index
    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS)

    # Get ripple filtered LFPs
    ripple_filtered_lfps = pd.DataFrame(
        filter_ripple_band(np.asarray(ripple_lfps)),
        index=ripple_lfps.index)

    # Get Ripple Times
    ripple_times = Kay_ripple_detector(
        time=ripple_filtered_lfps.index,
        filtered_lfps=ripple_filtered_lfps.values,
        speed=speed.values,
        sampling_frequency=LFP_SAMPLING_FREQUENCY,
        zscore_threshold=2.0,
        close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_labels(ripple_times, position_time)
    is_ripple = ripple_labels > 0
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    # Estmate ripple band power and change
    ripple_consensus_trace = pd.DataFrame(
        get_ripple_consensus_trace(
            ripple_filtered_lfps, LFP_SAMPLING_FREQUENCY),
        index=ripple_filtered_lfps.index,
        columns=['ripple_consensus_trace'])

    interpolation_method = 'linear'

    new_index = pd.Index(np.unique(np.concatenate(
        (ripple_consensus_trace.index, position_time))), name='time')
    ripple_consensus_trace = (ripple_consensus_trace
                              .reindex(index=new_index)
                              .interpolate(method=interpolation_method)
                              .reindex(index=position_time))
    ripple_consensus_trace_zscore = zscore(
        ripple_consensus_trace, nan_policy='omit')

    instantaneous_ripple_power = np.full_like(ripple_filtered_lfps, np.nan)
    not_null = np.all(pd.notnull(ripple_filtered_lfps), axis=1)
    instantaneous_ripple_power[not_null] = get_envelope(
        np.asarray(ripple_filtered_lfps)[not_null])**2
    instantaneous_ripple_power_change = np.nanmedian(
        instantaneous_ripple_power /
        np.nanmean(instantaneous_ripple_power, axis=0),
        axis=1)
    instantaneous_ripple_power_change = pd.DataFrame(
        instantaneous_ripple_power_change,
        index=ripple_filtered_lfps.index,
        columns=['instantaneous_ripple_power_change'])

    return dict(
        ripple_times=ripple_times,
        ripple_labels=ripple_labels,
        ripple_filtered_lfps=ripple_filtered_lfps,
        ripple_consensus_trace=ripple_consensus_trace,
        ripple_lfps=ripple_lfps,
        ripple_consensus_trace_zscore=ripple_consensus_trace_zscore,
        instantaneous_ripple_power_change=instantaneous_ripple_power_change,
        is_ripple=is_ripple)


def get_adhoc_multiunit(position_info, tetrode_keys, time_function,
                        position_to_linearize):
    time = position_info.index
    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, time_function)
    multiunit_spikes = (np.any(~np.isnan(multiunits.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY),
        index=position_info.index,
        columns=['firing_rate'])
    multiunit_rate_zscore = (
        (multiunit_firing_rate -
         np.nanmean(multiunit_firing_rate[position_info.nose_vel < 4])) /
        np.nanstd(multiunit_firing_rate[position_info.nose_vel < 4]))

    speed_feature = position_to_linearize[0].split('_')[0]
    speed = np.asarray(position_info[f'{speed_feature}_vel'])

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, speed,
        SAMPLING_FREQUENCY,
        minimum_duration=np.timedelta64(15, 'ms'), zscore_threshold=2.0,
        close_event_threshold=np.timedelta64(0, 'ms'))
    multiunit_high_synchrony_times.index = (
        multiunit_high_synchrony_times.index.rename('replay_number'))
    multiunit_high_synchrony_labels = get_labels(
        multiunit_high_synchrony_times, time)
    is_multiunit_high_synchrony = multiunit_high_synchrony_labels > 0
    multiunit_high_synchrony_times = multiunit_high_synchrony_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    return dict(
        multiunits=multiunits,
        multiunit_spikes=multiunit_spikes,
        multiunit_firing_rate=multiunit_firing_rate,
        multiunit_high_synchrony_times=multiunit_high_synchrony_times,
        multiunit_high_synchrony_labels=multiunit_high_synchrony_labels,
        multiunit_rate_zscore=multiunit_rate_zscore,
        is_multiunit_high_synchrony=is_multiunit_high_synchrony)


def load_data(epoch_key,
              position_to_linearize=['nose_x', 'nose_y'],
              max_distance_from_well=30,
              min_distance_traveled=50,
              ):
    logger.info('Loading position info...')
    environment = np.asarray(
        make_epochs_dataframe(ANIMALS).loc[epoch_key].environment)[0]
    if environment in ["lineartrack", "lineartack"]:
        edge_order, edge_spacing = LINEAR_EDGE_ORDER, LINEAR_EDGE_SPACING
    elif environment == "wtrack":
        edge_order, edge_spacing = WTRACK_EDGE_ORDER, WTRACK_EDGE_SPACING
    else:
        edge_order, edge_spacing = None, None
    position_info = get_interpolated_position_info(
        epoch_key,
        position_to_linearize=position_to_linearize,
        max_distance_from_well=max_distance_from_well,
        min_distance_traveled=min_distance_traveled,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    ).dropna(subset=["linear_position"])
    tetrode_info = make_tetrode_dataframe(
        ANIMALS, epoch_key=epoch_key)
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index

    logger.info('Loading multiunit...')

    def _time_function(*args, **kwargs):
        return position_info.index

    adhoc_multiunit = get_adhoc_multiunit(
        position_info, tetrode_keys, _time_function, position_to_linearize)

    logger.info('Loading spikes...')
    time = position_info.index
    try:
        neuron_info = make_neuron_dataframe(
            ANIMALS, exclude_animals=['Monty', 'Peanut']).xs(
                epoch_key, drop_level=False)
        neuron_info = neuron_info.loc[neuron_info.accepted.astype(bool)]
        spikes = get_all_spike_indicators(
            neuron_info.index, ANIMALS, _time_function).reindex(time)
    except (ValueError, KeyError):
        neuron_info = None
        spikes = None

    logger.info('Finding ripple times...')
    adhoc_ripple = get_adhoc_ripple(
        epoch_key, tetrode_info, time, position_to_linearize)

    track_graph = make_track_graph(epoch_key, ANIMALS)

    dio = get_DIO(epoch_key, ANIMALS)
    dio_indicator = get_DIO_indicator(
        epoch_key, ANIMALS, time_function=_time_function)
    theta = get_theta(
        tetrode_info, position_info)

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'neuron_info': neuron_info,
        'spikes': spikes,
        'dio': dio,
        'dio_indicator': dio_indicator,
        'track_graph': track_graph,
        'edge_order': edge_order,
        'edge_spacing': edge_spacing,
        **theta,
        **adhoc_ripple,
        **adhoc_multiunit,

    }


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'posdlc', 'posdlc')[epoch - 1]
    position_data = struct['data'][0, 0]
    field_names = struct['fields'][0, 0][0].split()
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')

    return pd.DataFrame(
        position_data[:, 1:], columns=field_names[1:], index=time)


def get_segments_df(epoch_key, animals, position_df, max_distance_from_well=30,
                    min_distance_traveled=50,
                    position_to_linearize=['nose_x', 'nose_y']):
    well_locations = get_well_locations(epoch_key, animals)
    position = position_df.loc[:, position_to_linearize].values
    segments_df, labeled_segments = segment_path(
        position_df.index, position, well_locations, epoch_key, animals,
        max_distance_from_well=max_distance_from_well)
    segments_df = score_inbound_outbound(
        segments_df, epoch_key, animals, min_distance_traveled)
    segments_df = segments_df.loc[
        :, ['from_well', 'to_well', 'task', 'is_correct', 'turn']]

    return segments_df, labeled_segments


def _get_linear_position_hmm(
    epoch_key, animals, position_df,
        max_distance_from_well=30,
        route_euclidean_distance_scaling=1,
        min_distance_traveled=50,
        sensor_std_dev=5,
        diagonal_bias=0.5,
        edge_order=WTRACK_EDGE_ORDER,
        edge_spacing=WTRACK_EDGE_SPACING,
        position_to_linearize=['nose_x', 'nose_y'],
        position_sampling_frequency=125):
    animal, day, epoch = epoch_key
    track_graph = make_track_graph(epoch_key, animals)
    position = np.asarray(position_df.loc[:, position_to_linearize])
    linearized_position_df = get_linearized_position(
        position=position,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
        use_HMM=False,
    )
    position_df = pd.concat(
        (position_df,
         linearized_position_df.set_index(position_df.index)), axis=1)
    try:
        SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                                  1.0: 'Left Arm',
                                  2.0: 'Right Arm',
                                  3.0: 'Left Arm',
                                  4.0: 'Right Arm'}
        position_df = position_df.assign(
            arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
        )
        segments_df, labeled_segments = get_segments_df(
            epoch_key, animals, position_df, max_distance_from_well,
            min_distance_traveled)

        segments_df = pd.merge(
            labeled_segments, segments_df, right_index=True,
            left_on='labeled_segments', how='outer')
        position_df = pd.concat((position_df, segments_df), axis=1)
        position_df['is_correct'] = position_df.is_correct.fillna(False)
    except ValueError:
        pass

    return position_df


def get_position_info(
    epoch_key, position_to_linearize=['nose_x', 'nose_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        skip_linearization=False, route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=0.5, position_sampling_frequency=125,
        edge_order=WTRACK_EDGE_ORDER, edge_spacing=WTRACK_EDGE_SPACING
):
    position_df = _get_pos_dataframe(epoch_key, ANIMALS)

    if not skip_linearization:
        position_df = _get_linear_position_hmm(
            epoch_key, ANIMALS, position_df,
            max_distance_from_well, route_euclidean_distance_scaling,
            min_distance_traveled, sensor_std_dev, diagonal_bias,
            edge_order=edge_order, edge_spacing=edge_spacing,
            position_to_linearize=position_to_linearize,
            position_sampling_frequency=position_sampling_frequency)

    return position_df

# max_distance_from_well=30 cms. This is perhaps ok for the tail but maybe the
# value needs to be lower for paws, nose etc.
# also eventually DIOs may help in the trajectory classification.


def get_interpolated_position_info(
    epoch_key, position_to_linearize=['nose_x', 'nose_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=1E-1, edge_order=WTRACK_EDGE_ORDER,
        edge_spacing=WTRACK_EDGE_SPACING):
    position_info = get_position_info(
        epoch_key, skip_linearization=True)
    position_info = position_info.resample('2ms').mean().interpolate('linear')

    position_info = _get_linear_position_hmm(
        epoch_key, ANIMALS, position_info,
        max_distance_from_well, route_euclidean_distance_scaling,
        min_distance_traveled, sensor_std_dev, diagonal_bias,
        edge_order=edge_order, edge_spacing=edge_spacing,
        position_to_linearize=position_to_linearize,
        position_sampling_frequency=500)

    return position_info


def get_sleep_and_prev_run_epochs():

    epoch_info = make_epochs_dataframe(ANIMALS)
    sleep_epoch_keys = []
    prev_run_epoch_keys = []

    for _, df in epoch_info.groupby(["animal", "day"]):
        is_w_track = np.asarray(df.iloc[:-1].environment.isin(["wtrack"]))

        is_sleep_after_run = np.asarray(
            (df.iloc[1:].type == "sleep") & is_w_track)
        sleep_ind = np.nonzero(is_sleep_after_run)[0] + 1

        sleep_epoch_keys.append(df.iloc[sleep_ind].index)
        prev_run_epoch_keys.append(df.iloc[sleep_ind - 1].index)

    sleep_epoch_keys = list(itertools.chain(*sleep_epoch_keys))
    prev_run_epoch_keys = list(itertools.chain(*prev_run_epoch_keys))

    return sleep_epoch_keys, prev_run_epoch_keys


def get_gnd_tetrode_filename(tetrode_key, animals):
    '''Returns a file name for the filtered LFP for an epoch.
    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.
    Returns
    -------
    filename : str
        File path to tetrode file LFP
    '''
    animal, day, epoch, tetrode_number = tetrode_key
    # add eeggnd to filename to get correct theta from a reference ntrode
    filename = (f'{animals[animal].short_name}eeggnd{day:02d}'
                f'-{epoch}-{tetrode_number:02d}.mat')
    return os.path.join(animals[animal].directory, 'EEG', filename)


def _get_thetafilter_kernel():
    '''Returns the pre-computed theta filter kernel from the Frank lab.
    The kernel is 150-250 Hz bandpass with 40 db roll off and 10 Hz
    sidebands. Sampling frequency is 1500 Hz.
    '''
    filter_file = '../src/thetafilter_5_11_1500.mat'
    thetafilter = loadmat(filter_file)
    return thetafilter['thetafilter_5_11_1500']['kernel'][0][0].flatten(), 1


def filter_theta_band(data):
    '''Returns a bandpass filtered signal between 150-250 Hz

    Parameters
    ----------
    data : array_like, shape (n_time,)

    Returns
    -------
    filtered_data : array_like, shape (n_time,)

    '''
    filter_numerator, filter_denominator = _get_thetafilter_kernel()
    is_nan = np.any(np.isnan(data), axis=-1)
    filtered_data = np.full_like(data, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filter_numerator, filter_denominator, data[~is_nan], axis=0)
    return filtered_data


def get_gnd_LFP_dataframe(tetrode_key, animals):
    '''Gets the LFP data for a given epoch and tetrode.

    Parameters
    ----------
    tetrode_key : tuple
        Unique key identifying the tetrode. Elements are
        (animal_short_name, day, epoch, tetrode_number).
    animals : dict of named-tuples
        Dictionary containing information about the directory for each
        animal. The key is the animal_short_name.

    Returns
    -------
    LFP : pandas dataframe
        Contains the electric potential and time
    '''
    try:
        lfp_file = loadmat(get_gnd_tetrode_filename(tetrode_key, ANIMALS))
        lfp_data = lfp_file['eeggnd'][0, -1][0, -1][0, -1]
        lfp_time = reconstruct_time(
            lfp_data['starttime'][0, 0].item(),
            lfp_data['data'][0, 0].size,
            float(lfp_data['samprate'][0, 0].squeeze()))
        return pd.Series(
            data=lfp_data['data'][0, 0].squeeze().astype(float),
            index=lfp_time,
            name='{0}_{1:02d}_{2:02}_{3:03}'.format(*tetrode_key))
    except (FileNotFoundError, TypeError):
        logger.warning('Failed to load file: {0}'.format(
            get_LFP_filename(tetrode_key, animals)))


def get_theta(tetrode_info, position_info):
    '''
    descending : more non-local
    ascending : more local/reverse
    '''
    tetrode_key = tetrode_info.loc[tetrode_info.suparea == 'cc'].index[0]

    gnd_lfp = get_gnd_LFP_dataframe(tetrode_key, ANIMALS)
    theta_filtered_lfp = filter_theta_band(gnd_lfp[:, np.newaxis]).squeeze()

    is_nan = np.isnan(theta_filtered_lfp)

    analytic_signal = np.full(theta_filtered_lfp.shape, np.nan, dtype=np.complex)
    analytic_signal[~is_nan] = hilbert(theta_filtered_lfp[~is_nan], axis=0)
    analytic_signal = pd.Series(analytic_signal, index=gnd_lfp.index)

    theta_phase = pd.Series(np.unwrap(np.angle(analytic_signal)), index=gnd_lfp.index)
    theta_power = np.abs(analytic_signal)**2

    new_index = pd.Index(
        np.unique(np.concatenate((position_info.index,
                                  gnd_lfp.index))),
        name="time"
    )


    theta_phase = (
        theta_phase.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=position_info.index)
    )

    theta_phase = (theta_phase + np.pi) % (2 * np.pi) - np.pi

    theta_power = (
        theta_power.reindex(index=new_index)
        .interpolate(method="linear")
        .reindex(index=position_info.index)
    )

    is_descending = (theta_phase >= 0.0)
    speed = position_info.nose_vel.values.squeeze()
    theta_power_zscore = (theta_power - np.nanmean(theta_power[speed >= 4])) / np.nanstd(theta_power[speed >= 4])

    return {
        'gnd_lfp': gnd_lfp,
        'theta_filtered_lfp': theta_filtered_lfp,
        'is_descending': is_descending,
        'theta_power': theta_power,
        'theta_phase': theta_phase,
        'theta_power_zscore': theta_power_zscore,
    }


def get_is_training(data, training_type='no_ripple_and_no_ascending_theta'):
    speed = np.asarray(data['position_info'].nose_vel).squeeze()
    not_ripple = ~np.asarray(data['is_ripple']).squeeze()
    not_ascending_theta = np.asarray(data['is_descending']).squeeze()
    training_types = {
        'all': np.ones_like(not_ripple),
        'no_ripple': not_ripple,
        'no_ripple_and_no_ascending_theta': (
            (not_ripple & (speed <= 4)) | not_ascending_theta & (speed > 4))
    }

    return training_types[training_type]
