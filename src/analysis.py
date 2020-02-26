
import networkx as nx
import numpy as np
from loren_frank_data_processing.track_segment_classification import (
    get_track_segments_from_graph, project_points_to_segment)


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
    map_edges = np.array(track_graph.edges)[track_segment_id]

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
    actual_edges = np.array(track_graph.edges)[track_segment_id]

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


def calculate_replay_distance(
        posterior, track_graph, decoder, position_2D, track_segment_id):
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
        replay_distance_from_animal_position.append(
            nx.shortest_path_length(copy_graph, source='actual_position',
                                    target='map_position', weight='distance'))
        copy_graph.remove_node('actual_position')
        copy_graph.remove_node('map_position')

    return np.asarray(replay_distance_from_animal_position)
