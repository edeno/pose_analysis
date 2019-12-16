import pandas as pd

from loren_frank_data_processing import (get_all_multiunit_indicators,
                                         make_tetrode_dataframe)
from loren_frank_data_processing.core import get_data_structure
from loren_frank_data_processing.position import (classify_track_segments,
                                                  make_track_graph)
from loren_frank_data_processing.track_segment_classification import \
    calculate_linear_distance
from src.parameters import ANIMALS


def load_data(epoch_key):
    position_info = get_interpolated_position_info(epoch_key)
    tetrode_info = make_tetrode_dataframe(
        ANIMALS).xs(epoch_key, drop_level=False)
    tetrode_keys = tetrode_info.loc[tetrode_info.area.isin(
        ['ca1R', 'ca1l'])].index

    def _time_function(*args, **kwargs):
        return position_info.index

    multiunits = get_all_multiunit_indicators(
        tetrode_keys, ANIMALS, _time_function)

    return {
        'position_info': position_info,
        'multiunits': multiunits
    }


def convert_linear_distance_to_linear_position(
        linear_distance, track_segment_id, edge_order, spacing=30):
    linear_position = linear_distance.copy()
    n_edges = len(edge_order)
    if isinstance(spacing, int) | isinstance(spacing, float):
        spacing = [spacing, ] * (n_edges - 1)

    for prev_edge, cur_edge, space in zip(
            edge_order[:-1], edge_order[1:], spacing):
        is_cur_edge = (track_segment_id == cur_edge)
        is_prev_edge = (track_segment_id == prev_edge)

        cur_distance = linear_position[is_cur_edge]
        cur_distance -= cur_distance.min()
        cur_distance += linear_position[is_prev_edge].max() + space
        linear_position[is_cur_edge] = cur_distance

    return linear_position


def _get_pos_dataframe(epoch_key, animals):
    animal, day, epoch = epoch_key
    struct = get_data_structure(
        animals[animal], day, 'posdlc', 'posdlc')[epoch - 1]
    position_data = struct['data'][0, 0]
    field_names = struct['fields'][0, 0][0].split(' ')
    time = pd.TimedeltaIndex(
        position_data[:, 0], unit='s', name='time')

    return pd.DataFrame(
        position_data[:, 1:], columns=field_names[1:], index=time)


def get_interpolated_position_info(epoch_key):
    position_info = _get_pos_dataframe(epoch_key, ANIMALS)

    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)
    position = position_info.loc[:, [
        'tailBase_x', 'tailBase_y']].values
    track_segment_id = classify_track_segments(
        track_graph, position,
        route_euclidean_distance_scaling=1E-1,
        sensor_std_dev=10)
    track_segment_id = pd.DataFrame(
        track_segment_id, index=position_info.index)
    position_info['linear_distance'] = calculate_linear_distance(
        track_graph, track_segment_id.values.squeeze(), center_well_id,
        position)

    position_info = position_info.resample('2ms').mean().interpolate('time')
    position_info.loc[
        position_info.linear_distance < 0, 'linear_distance'] = 0.0
    position_info['track_segment_id'] = (
        track_segment_id.reindex(index=position_info.index, method='pad'))

    EDGE_ORDER = [0, 1, 3, 2, 4]
    EDGE_SPACING = [0, 0, 0, 0]

    position_info['linear_position'] = convert_linear_distance_to_linear_position(
        position_info.linear_distance.values,
        position_info.track_segment_id.values, EDGE_ORDER,
        spacing=EDGE_SPACING)
    SEGMENT_ID_TO_ARM_NAME = {0.0: 'Center Arm',
                              1.0: 'Left Arm',
                              2.0: 'Right Arm',
                              3.0: 'Left Arm',
                              4.0: 'Right Arm'}
    position_info = position_info.assign(
        arm_name=lambda df: df.track_segment_id.map(SEGMENT_ID_TO_ARM_NAME)
    )

    return position_info
