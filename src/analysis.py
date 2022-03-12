import networkx as nx
import numpy as np
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)
import pandas as pd
import xarray as xr


def _get_MAP_estimate_2d_position_edges(posterior, track_graph, decoder):
    # Get 2D position on track from decoder MAP estimate
    map_position_ind = (
        posterior.where(decoder.is_track_interior_).argmax(
            "position", skipna=True).values
    )
    map_position_2d = decoder.place_bin_center_2D_position_[
        map_position_ind]

    # Figure out which track segment it belongs to
    track_segment_id = decoder.place_bin_center_ind_to_edge_id_[
        map_position_ind]
    map_edges = np.array(list(track_graph.edges))[track_segment_id]

    return map_position_2d, map_edges


def _get_animal_2d_projected_position_edges(
        track_graph, position_2D, track_segment_id):
    # Get animal's 2D position projected onto track
    track_segments = get_track_segments_from_graph(track_graph)
    projected_track_positions = project_points_to_segment(
        track_segments, position_2D)
    n_time = projected_track_positions.shape[0]
    actual_projected_position = projected_track_positions[(
        np.arange(n_time), track_segment_id)]

    # Add animal's position at time to track graph
    actual_edges = np.array(list(track_graph.edges))[track_segment_id]

    return actual_projected_position, actual_edges


def add_node(pos, edge, graph, node_name):
    node1, node2 = edge
    x3, y3 = pos

    x1, y1 = graph.nodes[node1]['pos']
    left_distance = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    nx.add_path(graph, [node1, node_name], distance=left_distance)

    x2, y2 = graph.nodes[node2]['pos']
    right_distance = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    nx.add_path(
        graph, [node_name, node2], distance=right_distance)

#


def calculate_replay_distance(
        posterior, track_graph, decoder, position_2D, track_segment_id):

    track_segment_id = np.asarray(track_segment_id).astype(int).squeeze()
    position_2D = np.asarray(position_2D)
    map_position_2d, map_edges = _get_MAP_estimate_2d_position_edges(
        posterior, track_graph, decoder)
    (actual_projected_position,
     actual_edges) = _get_animal_2d_projected_position_edges(
        track_graph, position_2D, track_segment_id)

    copy_graph = track_graph.copy()
    replay_distance_from_animal_position = []

    for actual_pos, actual_edge, map_pos, map_edge in zip(
            actual_projected_position, actual_edges, map_position_2d,
            map_edges):

        # Add actual position node
        add_node(actual_pos, actual_edge, copy_graph, 'actual_position')
        add_node(map_pos, map_edge, copy_graph, 'map_position')
        if np.all(actual_edge == map_edge):
            (x1, y1), (x2, y2) = actual_pos, map_pos
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            nx.add_path(
                copy_graph, ['actual_position', 'map_position'],
                distance=distance)

        replay_distance_from_animal_position.append(
            nx.shortest_path_length(copy_graph, source='actual_position',
                                    target='map_position', weight='distance'))
        copy_graph.remove_node('actual_position')
        copy_graph.remove_node('map_position')

    return np.asarray(replay_distance_from_animal_position)


def get_place_field_max(classifier):
    try:
        max_ind = classifier.place_fields_.argmax('position')
        return np.asarray(
            classifier.place_fields_.position[max_ind].values.tolist())
    except AttributeError:
        return np.asarray(
            [classifier.place_bin_centers_[gpi.argmax()]
             for gpi in classifier.ground_process_intensities_])


def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : ndarray, shape (n_time,)

    '''
    try:
        stacked_posterior = np.log(posterior_density.stack(
            z=['x_position', 'y_position']))
        map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
        map_estimate = np.asarray(map_estimate.values.tolist())
    except KeyError:
        map_estimate = posterior_density.position[
            np.log(posterior_density).argmax('position')]
        map_estimate = np.asarray(map_estimate)[:, np.newaxis]
    return map_estimate


def get_probability_of_state(results, posterior_type='acausal_posterior'):
    fragmented = (results[posterior_type]
                  .sel(state=['Inbound-Fragmented', 'Outbound-Fragmented'])
                  .sum(['state', 'position'])
                  .assign_coords({'state': 'Fragmented'}))
    probability = (results[posterior_type]
                   .sum('position')
                   .drop_sel(state=['Inbound-Fragmented', 'Outbound-Fragmented']))
    return xr.concat((probability, fragmented), dim='state')


def classify_states(probability, probability_threshold=0.8, sampling_frequency=500):
    is_classified = (probability > probability_threshold).sum('state').astype(bool)
    max_state = probability.idxmax('state')
    classified_states_by_time = max_state.isel(time=is_classified)

    indexes = np.unique(classified_states_by_time.values, return_index=True)[1]
    classified_states = classified_states_by_time.values[sorted(indexes)]
    is_state = (probability > probability_threshold).sum('time') > 0
    state_duration = (probability > probability_threshold).sum('time') / sampling_frequency
    return classified_states_by_time, classified_states, is_state.values, state_duration.values


def get_replay_info(data, results, epoch_key):
    classified_states = []
    is_state = []
    state_duration = []

    for ripple_number in data['ripple_times'].index:
        ripple = data['ripple_times'].loc[ripple_number]

        start_time = ripple.start_time
        end_time = ripple.end_time

        probability = get_probability_of_state(
                        results.sel(time=slice(start_time / np.timedelta64(1, 's'),
                                               end_time / np.timedelta64(1, 's'))))
        _, classified_states_temp, is_state_temp, state_duration_temp = classify_states(probability)
        classified_states.append(classified_states_temp)
        is_state.append(is_state_temp)
        state_duration.append(state_duration_temp)
        
    is_state = pd.DataFrame(np.stack(is_state),
                            columns=probability.state,
                            index=data['ripple_times'].index)
    is_state['Animal'] = epoch_key[0]
    is_state['Day'] = epoch_key[1]
    is_state['Epoch'] = epoch_key[2]
    
    state_duration = pd.DataFrame(np.stack(state_duration),
                                  columns=probability.state + '_duration',
                                  index=data['ripple_times'].index)
    replay_info = pd.concat((is_state, state_duration), axis=1)
    replay_info = replay_info.reset_index().set_index(['Animal', 'Day', 'Epoch', 'replay_number'])
    
    return replay_info, classified_states