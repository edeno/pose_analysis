from logging import getLogger

import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         get_all_spike_indicators, get_LFPs,
                                         get_trial_time, make_neuron_dataframe,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import get_data_structure
from loren_frank_data_processing.position import (_calulcate_linear_position,
                                                  calculate_linear_velocity,
                                                  get_well_locations,
                                                  make_track_graph)
from loren_frank_data_processing.track_segment_classification import (
    calculate_linear_distance, classify_track_segments)
from loren_frank_data_processing.well_traversal_classification import (
    score_inbound_outbound, segment_path)
from ripple_detection import (Kay_ripple_detector, filter_ripple_band,
                              get_multiunit_population_firing_rate,
                              multiunit_HSE_detector)
from ripple_detection.core import gaussian_smooth
from spectral_connectivity import Connectivity, Multitaper
from src.parameters import (ANIMALS, EDGE_ORDER, EDGE_SPACING,
                            SAMPLING_FREQUENCY)

logger = getLogger(__name__)


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


def get_adhoc_ripple(epoch_key, tetrode_info, position_time):
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
    speed = position_df['tailBase_vel']

    # Load LFPs
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index
    ripple_lfps = get_LFPs(tetrode_keys, ANIMALS).reindex(time)

    # Get ripple filtered LFPs
    ripple_filtered_lfps = pd.DataFrame(
        np.stack([filter_ripple_band(
            ripple_lfps.values[:, ind],
            sampling_frequency=LFP_SAMPLING_FREQUENCY)
            for ind in np.arange(ripple_lfps.shape[1])], axis=1),
        index=ripple_lfps.index)

    # Get Ripple Times
    ripple_times = Kay_ripple_detector(
        time, ripple_lfps.values, speed.values, LFP_SAMPLING_FREQUENCY,
        zscore_threshold=2.0, close_ripple_threshold=np.timedelta64(0, 'ms'),
        minimum_duration=np.timedelta64(15, 'ms'))

    ripple_times.index = ripple_times.index.rename('replay_number')
    ripple_labels = get_labels(ripple_times, position_time)
    is_ripple = ripple_labels > 0
    ripple_times = ripple_times.assign(
        duration=lambda df: (df.end_time - df.start_time).dt.total_seconds())

    # Estmate ripple band power and change
    consensus_ripple_trace = np.sum(ripple_filtered_lfps ** 2, axis=1)
    consensus_ripple_trace = gaussian_smooth(
        consensus_ripple_trace, 0.004, LFP_SAMPLING_FREQUENCY)
    consensus_ripple_trace = np.sqrt(consensus_ripple_trace)
    consensus_ripple_trace = pd.DataFrame(
        {"consensus_ripple_trace": consensus_ripple_trace},
        index=ripple_lfps.index)

    ripple_power = estimate_ripple_band_power(
        ripple_lfps, LFP_SAMPLING_FREQUENCY)
    interpolated_ripple_power = ripple_power.interpolate()

    ripple_power_change = interpolated_ripple_power.transform(
        lambda df: df / df.mean())
    ripple_power_zscore = np.log(interpolated_ripple_power).transform(
        lambda df: (df - df.mean()) / df.std())

    return dict(ripple_times=ripple_times,
                ripple_labels=ripple_labels,
                ripple_filtered_lfps=ripple_filtered_lfps,
                ripple_power=ripple_power,
                ripple_lfps=ripple_lfps,
                ripple_power_change=ripple_power_change,
                ripple_power_zscore=ripple_power_zscore,
                is_ripple=is_ripple,
                consensus_ripple_trace=consensus_ripple_trace,
                )


def get_adhoc_multiunit(position_info, tetrode_keys, time_function):
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
    multiunit_rate_change = multiunit_firing_rate.transform(
        lambda df: df / df.mean())
    multiunit_rate_zscore = np.log(multiunit_firing_rate).transform(
        lambda df: (df - df.mean()) / df.std())

    multiunit_high_synchrony_times = multiunit_HSE_detector(
        time, multiunit_spikes, position_info.tailBase_vel.values,
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
        multiunit_rate_change=multiunit_rate_change,
        multiunit_rate_zscore=multiunit_rate_zscore,
        is_multiunit_high_synchrony=is_multiunit_high_synchrony)


def load_data(epoch_key,
              position_to_linearize=['tailBase_x', 'tailBase_y'],
              max_distance_from_well=30,
              min_distance_traveled=50,
              ):
    logger.info('Loading position info...')
    position_info = get_interpolated_position_info(
        epoch_key,
        position_to_linearize=position_to_linearize,
        max_distance_from_well=max_distance_from_well,
        min_distance_traveled=min_distance_traveled,
    ).dropna(subset=["linear_position"])
    tetrode_info = make_tetrode_dataframe(
        ANIMALS, epoch_key=epoch_key)
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index

    logger.info('Loading multiunit...')

    def _time_function(*args, **kwargs):
        return position_info.index

    adhoc_multiunit = get_adhoc_multiunit(
        position_info, tetrode_keys, _time_function)

    logger.info('Loading spikes...')
    time = position_info.index
    neuron_info = make_neuron_dataframe(ANIMALS).xs(
        epoch_key, drop_level=False)
    neuron_info = neuron_info.loc[neuron_info.accepted.astype(bool)]
    spikes = get_all_spike_indicators(
        neuron_info.index, ANIMALS, _time_function).reindex(time)

    logger.info('Finding ripple times...')
    adhoc_ripple = get_adhoc_ripple(epoch_key, tetrode_info, time)

    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)

    return {
        'position_info': position_info,
        'tetrode_info': tetrode_info,
        'neuron_info': neuron_info,
        'spikes': spikes,
        'track_graph': track_graph,
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
                    position_to_linearize=['tailBase_x', 'tailBase_y']):
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
        edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING,
        position_to_linearize=['tailBase_x', 'tailBase_y'],
        position_sampling_frequency=125):
    animal, day, epoch = epoch_key
    track_graph, center_well_id = make_track_graph(epoch_key, animals)
    position = position_df.loc[:, position_to_linearize].values
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=route_euclidean_distance_scaling,
        sensor_std_dev=sensor_std_dev,
        diagonal_bias=diagonal_bias)
    (position_df['linear_distance'],
     position_df['projected_x_position'],
     position_df['projected_y_position']) = calculate_linear_distance(
        track_graph, track_segment_id, center_well_id, position)
    position_df['track_segment_id'] = track_segment_id
    SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                              1.0: 'Left Arm',
                              2.0: 'Right Arm',
                              3.0: 'Left Arm',
                              4.0: 'Right Arm'}
    position_df = position_df.assign(
        arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
    )
    try:
        segments_df, labeled_segments = get_segments_df(
            epoch_key, animals, position_df, max_distance_from_well,
            min_distance_traveled)

        segments_df = pd.merge(
            labeled_segments, segments_df, right_index=True,
            left_on='labeled_segments', how='outer')
        position_df = pd.concat((position_df, segments_df), axis=1)
        position_df['linear_position'] = _calulcate_linear_position(
            position_df.linear_distance.values,
            position_df.track_segment_id.values, track_graph, center_well_id,
            edge_order=edge_order, edge_spacing=edge_spacing)
        position_df['is_correct'] = position_df.is_correct.fillna(False)
    except TypeError:
        position_df['linear_position'] = position_df['linear_distance'].copy()
    position_df['linear_velocity'] = calculate_linear_velocity(
        position_df.linear_distance, smooth_duration=0.500,
        sampling_frequency=position_sampling_frequency)
    position_df['linear_speed'] = np.abs(position_df.linear_velocity)

    return position_df


def get_position_info(
    epoch_key, position_to_linearize=['tailBase_x', 'tailBase_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        skip_linearization=False, route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=0.5, position_sampling_frequency=125,
):
    position_df = _get_pos_dataframe(epoch_key, ANIMALS)

    if not skip_linearization:
        position_df = _get_linear_position_hmm(
            epoch_key, ANIMALS, position_df,
            max_distance_from_well, route_euclidean_distance_scaling,
            min_distance_traveled, sensor_std_dev, diagonal_bias,
            edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING,
            position_to_linearize=position_to_linearize,
            position_sampling_frequency=position_sampling_frequency)

    return position_df

# max_distance_from_well=30 cms. This is perhaps ok for the tail but maybe the
# value needs to be lower for paws, nose etc.
# also eventually DIOs may help in the trajectory classification.


def get_interpolated_position_info(
    epoch_key, position_to_linearize=['tailBase_x', 'tailBase_y'],
        max_distance_from_well=30, min_distance_traveled=50,
        route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=5, diagonal_bias=1E-1):
    position_info = get_position_info(
        epoch_key, skip_linearization=True)
    position_info = position_info.resample('2ms').mean().interpolate('linear')

    position_info = _get_linear_position_hmm(
        epoch_key, ANIMALS, position_info,
        max_distance_from_well, route_euclidean_distance_scaling,
        min_distance_traveled, sensor_std_dev, diagonal_bias,
        edge_order=EDGE_ORDER, edge_spacing=EDGE_SPACING,
        position_to_linearize=position_to_linearize,
        position_sampling_frequency=500)

    return position_info
