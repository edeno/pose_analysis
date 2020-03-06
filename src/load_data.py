import numpy as np
import pandas as pd
from loren_frank_data_processing import (get_all_multiunit_indicators,
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
from ripple_detection import get_multiunit_population_firing_rate

from src.parameters import (ANIMALS, EDGE_ORDER, EDGE_SPACING,
                            SAMPLING_FREQUENCY)


def load_data(epoch_key, position_to_linearize=['tail_x', 'tail_y']):
    position_info = get_interpolated_position_info(
        epoch_key, position_to_linearize)
    tetrode_info = make_tetrode_dataframe(
        ANIMALS, epoch_key=epoch_key)
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1L'])].index

    def _time_function(*args, **kwargs):
        return position_info.index

    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, _time_function)
    multiunit_spikes = (np.any(~np.isnan(multiunits.values), axis=1)
                        ).astype(np.float)
    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(
            multiunit_spikes, SAMPLING_FREQUENCY), index=position_info.index,
        columns=['firing_rate'])

    return {
        'position_info': position_info,
        'multiunits': multiunits,
        'multiunit_firing_rate': multiunit_firing_rate,
        'tetrode_info': tetrode_info,
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


def get_segments_df(epoch_key, animals, position_df, max_distance_from_well=5,
                    min_distance_traveled=50,
                    position_to_linearize=['tail_x', 'tail_y']):
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
        position_to_linearize=['tail_x', 'tail_y'],
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
    epoch_key, position_to_linearize=['tail_x', 'tail_y'],
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

# max_distance_from_well=30 cms. This is perhaps ok for the tail but maybe the value needs to be lower for paws, nose etc. 
# also eventually DIOs may help in the trajectory classification. 
def get_interpolated_position_info(
    epoch_key, position_to_linearize=['tail_x', 'tail_y'],
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
