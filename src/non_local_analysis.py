import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing import make_epochs_dataframe
from replay_identification import ClusterlessDetector
from ripple_detection.core import (exclude_movement, gaussian_smooth,
                                   segment_boolean_series)
from src.load_data import get_labels, load_data
from src.parameters import ANIMALS, PROCESSED_DATA_DIR
from tqdm.auto import tqdm
from trajectory_analysis_tools import (get_ahead_behind_distance,
                                       get_highest_posterior_threshold,
                                       get_HPD_spatial_coverage,
                                       get_trajectory_data)

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level='INFO', format=FORMAT, datefmt='%d-%b-%y %H:%M:%S')


def get_map_speed(
    posterior,
    track_graph1,
    place_bin_center_ind_to_node,
    dt,
):
    map_position_ind = np.argmax(posterior, axis=1)
    node_ids = place_bin_center_ind_to_node[map_position_ind]
    n_time = len(node_ids)

    if n_time == 1:
        return np.asarray([np.nan])
    elif n_time == 2:
        speed = np.asarray([])
        speed = np.insert(
            speed,
            0,
            nx.shortest_path_length(
                track_graph1, source=node_ids[0], target=node_ids[1],
                weight="distance",
            )
            / dt,
        )
        speed = np.insert(
            speed,
            -1,
            nx.shortest_path_length(
                track_graph1, source=node_ids[-2], target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    else:
        speed = []
        for node1, node2 in zip(node_ids[:-2], node_ids[2:]):
            speed.append(
                nx.shortest_path_length(
                    track_graph1, source=node1, target=node2,
                    weight="distance",
                )
                / (2.0 * dt)
            )
        speed = np.asarray(speed)
        speed = np.insert(
            speed,
            0,
            nx.shortest_path_length(
                track_graph1, source=node_ids[0], target=node_ids[1],
                weight="distance",
            )
            / dt,
        )
        speed = np.insert(
            speed,
            -1,
            nx.shortest_path_length(
                track_graph1, source=node_ids[-2], target=node_ids[-1],
                weight="distance",
            )
            / dt,
        )
    return np.abs(speed)


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


def make_non_local_info(training_type='no_ripple', prob_threshold=0.80):

    sampling_frequency = 500
    SMOOTH_SIGMA = 0.0025

    epoch_info = make_epochs_dataframe(ANIMALS)
    epoch_info = epoch_info.loc[(epoch_info.type == 'run')]

    animal = []
    day = []
    epoch = []
    event_number = []
    environment = []
    duration = []
    median_ripple_zscore = []
    median_multiunit_zscore = []
    median_speed = []
    median_hpd_pct = []
    is_swr_overlap = []
    is_mua_overlap = []
    pct_overlap_swr = []
    pct_overlap_mua = []
    median_pct_distance_from_animal = []
    median_replay_speed = []
    pct_participating_tetrodes = []
    n_participating_tetrodes = []
    median_pct_participating_tetrodes = []
    median_n_participating_tetrodes = []

    swr_start_time_offset = []
    swr_end_time_offset = []
    mua_start_time_offset = []
    mua_end_time_offset = []

    non_local_time_outside_swr = []
    pct_non_local_time_outside_swr = []
    local_time_inside_swr = []
    pct_local_time_inside_swr = []

    non_local_time_outside_mua = []
    pct_non_local_time_outside_mua = []
    local_time_inside_mua = []
    pct_local_time_inside_mua = []

    for epoch_key in tqdm(epoch_info.index):
        logging.info(epoch_key)
        try:
            epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
            results_filename = os.path.join(
                PROCESSED_DATA_DIR,
                f"{epoch_identifier}_clusterless_non_local_{training_type}.nc"
            )

            data = load_data(epoch_key,
                             position_to_linearize=['nose_x', 'nose_y'],
                             max_distance_from_well=30,
                             min_distance_traveled=50,
                             )
            results = xr.open_dataset(results_filename)

            is_training = get_is_training(data, training_type=training_type)
            detector_parameters = dict(
                track_graph=data['track_graph'],
                edge_order=data['edge_order'],
                edge_spacing=data['edge_spacing'],
                clusterless_algorithm='multiunit_likelihood_integer',
                clusterless_algorithm_params=dict(
                    mark_std=24.0, position_std=6.0, block_size=100),
            )
            detector = ClusterlessDetector(**detector_parameters)
            fit_args = dict(
                is_training=is_training,
                position=data['position_info'].linear_position,
                multiunits=data['multiunits'],
            )
            detector.fit(**fit_args)

            non_local_times = exclude_movement(
                segment_boolean_series(
                    (results.non_local_probability > prob_threshold
                     ).to_pandas()),
                speed=data['position_info'].nose_vel.values,
                time=results.time)
            n_non_local = len(non_local_times)
            non_local_times = pd.DataFrame(
                non_local_times, index=pd.Index(np.arange(n_non_local) + 1,
                                                name='event_number'),
                columns=['start_time', 'end_time'])
            non_local_labels = get_labels(non_local_times, results.time)

            logging.info('Computing non-local statistics...')

            is_non_local = non_local_labels.values > 0
            is_local = np.squeeze(results.non_local_probability.values < (1 - prob_threshold))

            non_local_time_outside_swr.append(n_non_local * [
                (~data['is_ripple'].values & is_non_local).sum() /
                sampling_frequency])
            pct_non_local_time_outside_swr.append(n_non_local * [
                100 *
                (~data['is_ripple'].values & is_non_local).sum() /
                is_non_local.sum()])
            local_time_inside_swr.append(n_non_local * [
                (data['is_ripple'].values.squeeze() & is_local).sum() /
                sampling_frequency])
            pct_local_time_inside_swr.append(n_non_local * [
                100 *
                (data['is_ripple'].values.squeeze() & is_local).sum() /
                data['is_ripple'].values.sum()])

            non_local_time_outside_mua.append(n_non_local * [
                (~data['is_multiunit_high_synchrony'].values & is_non_local).sum() /
                sampling_frequency])
            pct_non_local_time_outside_mua.append(n_non_local * [
                100 *
                (~data['is_multiunit_high_synchrony'].values & is_non_local).sum() /
                is_non_local.sum()])
            local_time_inside_mua.append(n_non_local * [
                (data['is_multiunit_high_synchrony'].values.squeeze() & is_local).sum() /
                sampling_frequency])
            pct_local_time_inside_mua.append(n_non_local * [
                100 *
                (data['is_multiunit_high_synchrony'].values.squeeze() & is_local).sum() /
                data['is_multiunit_high_synchrony'].values.sum()])

            for non_local_event_number in non_local_times.index:
                is_non_local_event = (non_local_labels.values.squeeze() ==
                                      non_local_event_number)

                animal.append(epoch_key[0])
                day.append(epoch_key[1])
                epoch.append(epoch_key[2])
                event_number.append(non_local_event_number)
                environment.append(
                    str(epoch_info.loc[epoch_key].environment.values[0]))

                duration.append(is_non_local_event.sum() / sampling_frequency)

                median_ripple_zscore.append(np.nanmedian(
                    data['ripple_consensus_trace_zscore'].values[is_non_local_event]))
                median_multiunit_zscore.append(np.nanmedian(
                    data['multiunit_rate_zscore'].values[is_non_local_event]))
                median_speed.append(np.nanmedian(
                    data['position_info'].nose_vel.values.squeeze()[is_non_local_event]))

                posterior = results.isel(
                    time=is_non_local_event).acausal_posterior.sum('state')

                max_coverage = np.sum(
                    np.diff(detector.place_bin_edges_.squeeze())[
                        detector.is_track_interior_])
                median_hpd_pct.append(
                    np.nanmedian(100 *
                                 get_HPD_spatial_coverage(
                                     posterior,
                                     get_highest_posterior_threshold(
                                         posterior, coverage=0.95)
                                 ) /
                                 max_coverage))

                is_swr_overlap.append(
                    np.any(data['is_ripple'].iloc[is_non_local_event]))
                is_mua_overlap.append(
                    np.any(data['is_multiunit_high_synchrony'].iloc[is_non_local_event]))
                pct_overlap_swr.append(
                    100 *
                    data['is_ripple'].iloc[is_non_local_event].values.sum() /
                    is_non_local_event.sum())
                pct_overlap_mua.append(
                    100 *
                    data['is_multiunit_high_synchrony'].iloc[is_non_local_event].values.sum()
                    / is_non_local_event.sum())

                trajectory_data = get_trajectory_data(
                    posterior=posterior,
                    track_graph=data['track_graph'],
                    decoder=detector,
                    actual_projected_position=data['position_info'][[
                        'projected_x_position', 'projected_y_position']],
                    track_segment_id=data['position_info'].track_segment_id,
                    actual_orientation=data['position_info'].body_dir,
                )
                max_distance = float(
                    np.max(data['position_info'].linear_position.values)
                    - np.sum(detector.edge_spacing))
                median_pct_distance_from_animal.append(
                    np.nanmedian(100 * np.abs(get_ahead_behind_distance(
                        data['track_graph'], *trajectory_data)) / max_distance))

                median_replay_speed.append(np.nanmedian(gaussian_smooth(
                    get_map_speed(
                        posterior.values,
                        detector.track_graph_with_bin_centers_edges_,
                        detector.place_bin_centers_nodes_df_.node_id.values,
                        1 / sampling_frequency,
                    ), SMOOTH_SIGMA, sampling_frequency)) / 100)

                n_participating_tetrodes.append(int((
                    (~np.isnan(data['multiunits'].isel(
                        time=is_non_local_event)))
                    .sum(['features', 'time']) > 0)
                    .sum('tetrodes').values))
                pct_participating_tetrodes.append(
                    100 * n_participating_tetrodes[-1] /
                    len(data['multiunits'].tetrodes))
                median_pct_participating_tetrodes.append(
                    100 * np.nanmedian((
                        (~np.isnan(data['multiunits'].isel(
                            time=is_non_local_event)))
                        .sum('features') > 0)
                        .mean('tetrodes')))
                median_n_participating_tetrodes.append(
                    np.nanmedian((
                        (~np.isnan(data['multiunits'].isel(
                            time=is_non_local_event)))
                        .sum('features') > 0)
                        .sum('tetrodes')))

                swr_overlap_numbers = np.unique(
                    data['ripple_labels'].iloc[is_non_local_event])
                swr_overlap_numbers = swr_overlap_numbers[
                    swr_overlap_numbers > 0]

                overlap_swr_start_times = np.asarray(
                    data['ripple_times'].loc[swr_overlap_numbers].start_time /
                    np.timedelta64(1, 's'))
                overlap_swr_end_times = np.asarray(
                    data['ripple_times'].loc[swr_overlap_numbers].end_time /
                    np.timedelta64(1, 's'))

                mua_overlap_numbers = np.unique(
                    data['multiunit_high_synchrony_labels']
                    .iloc[is_non_local_event])
                mua_overlap_numbers = mua_overlap_numbers[
                    mua_overlap_numbers > 0]

                overlap_mua_start_times = np.asarray(
                    data['multiunit_high_synchrony_times'].loc[mua_overlap_numbers].start_time /
                    np.timedelta64(1, 's'))
                overlap_mua_end_times = np.asarray(
                    data['multiunit_high_synchrony_times'].loc[mua_overlap_numbers].end_time /
                    np.timedelta64(1, 's'))

                non_local_event_start_time = np.asarray(
                    results.time[is_non_local_event][0])
                non_local_event_end_time = np.asarray(
                    results.time[is_non_local_event][-1])

                swr_start_time_offset.append(
                    non_local_event_start_time - overlap_swr_start_times)
                swr_end_time_offset.append(
                    non_local_event_end_time - overlap_swr_end_times)

                mua_start_time_offset.append(
                    non_local_event_start_time - overlap_mua_start_times)
                mua_end_time_offset.append(
                    non_local_event_end_time - overlap_mua_end_times)

        except Exception as e:
            logging.warning(e)

    return pd.DataFrame(
        {"animal": animal,
         "day": day,
         "epoch": epoch,
         "event_number": event_number,
         "environment": environment,
         "duration": duration,
         "median_ripple_zscore": median_ripple_zscore,
         "median_multiunit_zscore": median_multiunit_zscore,
         "median_speed": median_speed,
         "median_hpd_pct": median_hpd_pct,
         "is_swr_overlap": is_swr_overlap,
         "is_mua_overlap": is_mua_overlap,
         "pct_overlap_swr": pct_overlap_swr,
         "pct_overlap_mua": pct_overlap_mua,
         "median_pct_distance_from_animal": median_pct_distance_from_animal,
         "median_replay_speed": median_replay_speed,
         "pct_participating_tetrodes": pct_participating_tetrodes,
         "n_participating_tetrodes": n_participating_tetrodes,
         "median_pct_participating_tetrodes": median_pct_participating_tetrodes,
         "median_n_participating_tetrodes": median_n_participating_tetrodes,
         "swr_start_time_offset": swr_start_time_offset,
         "swr_end_time_offset": swr_end_time_offset,
         "mua_start_time_offset": mua_start_time_offset,
         "mua_end_time_offset": mua_end_time_offset,
         "non_local_time_outside_swr": np.concatenate(
             non_local_time_outside_swr),
         "pct_non_local_time_outside_swr": np.concatenate(
             pct_non_local_time_outside_swr),
         "local_time_inside_swr": np.concatenate(local_time_inside_swr),
         "pct_local_time_inside_swr": np.concatenate(
             pct_local_time_inside_swr),
         "non_local_time_outside_mua": np.concatenate(
             non_local_time_outside_mua),
         "pct_non_local_time_outside_mua": np.concatenate(
             pct_non_local_time_outside_mua),
         "local_time_inside_mua": np.concatenate(local_time_inside_mua),
         "pct_local_time_inside_mua": np.concatenate(
             pct_local_time_inside_mua),
         }
    ).set_index(['animal', 'day', 'epoch', 'event_number'])


def non_local_stats_by_epoch(
        epoch_key, training_type, data, results, epoch_info,
        prob_threshold, sampling_frequency):

    try:
        is_training = get_is_training(
            data, training_type=training_type)
        detector_parameters = dict(
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
            clusterless_algorithm='multiunit_likelihood_integer',
            clusterless_algorithm_params=dict(
                mark_std=24.0, position_std=6.0, block_size=100),
        )
        detector = ClusterlessDetector(**detector_parameters)
        fit_args = dict(
            is_training=is_training,
            position=data['position_info'].linear_position,
            multiunits=data['multiunits'],
        )
        detector.fit(**fit_args)

        non_local_times = exclude_movement(
            segment_boolean_series(
                (results.non_local_probability > prob_threshold
                 ).to_pandas()),
            speed=data['position_info'].nose_vel.values,
            time=results.time)
        n_non_local = len(non_local_times)
        non_local_times = pd.DataFrame(
            non_local_times, index=pd.Index(np.arange(n_non_local) + 1,
                                            name='event_number'),
            columns=['start_time', 'end_time'])
        non_local_labels = get_labels(non_local_times, results.time)

        logging.info('Computing non-local statistics...')
        return compute_non_local_stats(
            epoch_key, data, epoch_info, results, detector,
            non_local_times, non_local_labels,
            sampling_frequency)
    except Exception as e:
        logging.warning(e)


def compute_non_local_stats(epoch_key, data, epoch_info, results, detector,
                            non_local_times, non_local_labels,
                            sampling_frequency):
    SMOOTH_SIGMA = 0.0025

    n_non_local = len(non_local_times)
    is_non_local = non_local_labels.values > 0

    animal = []
    day = []
    epoch = []
    event_number = []
    environment = []
    duration = []
    median_ripple_zscore = []
    median_multiunit_zscore = []
    median_speed = []
    median_hpd_pct = []
    is_swr_overlap = []
    is_mua_overlap = []
    pct_overlap_swr = []
    pct_overlap_mua = []
    median_pct_distance_from_animal = []
    median_replay_speed = []
    pct_participating_tetrodes = []
    n_participating_tetrodes = []
    median_pct_participating_tetrodes = []
    median_n_participating_tetrodes = []

    swr_start_time_offset = []
    swr_end_time_offset = []
    mua_start_time_offset = []
    mua_end_time_offset = []

    non_local_time_outside_swr = []
    pct_non_local_time_outside_swr = []
    local_time_inside_swr = []
    pct_local_time_inside_swr = []

    non_local_time_outside_mua = []
    pct_non_local_time_outside_mua = []
    local_time_inside_mua = []
    pct_local_time_inside_mua = []

    non_local_time_outside_swr.append(n_non_local * [
        (~data['is_ripple'].values & is_non_local).sum() /
        sampling_frequency])
    pct_non_local_time_outside_swr.append(n_non_local * [
        100 *
        (~data['is_ripple'].values & is_non_local).sum() /
        is_non_local.sum()])
    local_time_inside_swr.append(n_non_local * [
        (data['is_ripple'].values & ~is_non_local).sum() /
        sampling_frequency])
    pct_local_time_inside_swr.append(n_non_local * [
        100 *
        (data['is_ripple'].values & ~is_non_local).sum() /
        data['is_ripple'].values.sum()])

    non_local_time_outside_mua.append(n_non_local * [
        (~data['is_multiunit_high_synchrony'].values & is_non_local).sum() /
        sampling_frequency])
    pct_non_local_time_outside_mua.append(n_non_local * [
        100 *
        (~data['is_multiunit_high_synchrony'].values & is_non_local).sum() /
        is_non_local.sum()])
    local_time_inside_mua.append(n_non_local * [
        (data['is_multiunit_high_synchrony'].values & ~is_non_local).sum() /
        sampling_frequency])
    pct_local_time_inside_mua.append(n_non_local * [
        100 *
        (data['is_multiunit_high_synchrony'].values & ~is_non_local).sum() /
        data['is_multiunit_high_synchrony'].values.sum()])

    for non_local_event_number in non_local_times.index:
        is_non_local_event = (non_local_labels.values.squeeze() ==
                              non_local_event_number)

        animal.append(epoch_key[0])
        day.append(epoch_key[1])
        epoch.append(epoch_key[2])
        event_number.append(non_local_event_number)
        environment.append(
            str(epoch_info.loc[epoch_key].environment.values[0]))

        duration.append(
            is_non_local_event.sum() / sampling_frequency)
        median_ripple_zscore.append(np.nanmedian(
            data['ripple_consensus_trace_zscore'].values[is_non_local_event]))
        median_multiunit_zscore.append(np.nanmedian(
            data['multiunit_rate_zscore'].values[is_non_local_event]))
        median_speed.append(np.nanmedian(
            data['position_info'].nose_vel.values.squeeze()[is_non_local_event]))

        posterior = results.isel(
            time=is_non_local_event).acausal_posterior.sum('state')
        max_coverage = np.sum(
            np.diff(detector.place_bin_edges_.squeeze())[
                detector.is_track_interior_])
        median_hpd_pct.append(
            np.nanmedian(100 *
                         get_HPD_spatial_coverage(
                             posterior,
                             get_highest_posterior_threshold(
                                 posterior, coverage=0.95)
                         ) /
                         max_coverage))
        is_swr_overlap.append(
            np.any(data['is_ripple'].iloc[is_non_local_event]))
        is_mua_overlap.append(
            np.any(data['is_multiunit_high_synchrony'].iloc[is_non_local_event]))
        pct_overlap_swr.append(
            100 *
            data['is_ripple'].iloc[is_non_local_event].values.sum() /
            is_non_local_event.sum())
        pct_overlap_mua.append(
            100 *
            data['is_multiunit_high_synchrony'].iloc[is_non_local_event].values.sum()
            / is_non_local_event.sum())

        trajectory_data = get_trajectory_data(
            posterior=posterior,
            track_graph=data['track_graph'],
            decoder=detector,
            actual_projected_position=data['position_info'][[
                'projected_x_position', 'projected_y_position']],
            track_segment_id=data['position_info'].track_segment_id,
            actual_orientation=data['position_info'].body_dir,
        )
        max_distance = float(
            np.max(data['position_info'].linear_position.values)
            - np.sum(detector.edge_spacing))
        median_pct_distance_from_animal.append(
            np.nanmedian(100 * np.abs(get_ahead_behind_distance(
                data['track_graph'], *trajectory_data)) / max_distance))

        median_replay_speed.append(np.nanmedian(gaussian_smooth(
            get_map_speed(
                posterior.values,
                detector.track_graph_with_bin_centers_edges_,
                detector.place_bin_centers_nodes_df_.node_id.values,
                1 / sampling_frequency,
            ), SMOOTH_SIGMA, sampling_frequency)) / 100)

        n_participating_tetrodes.append(int((
            (~np.isnan(data['multiunits'].isel(
                time=is_non_local_event)))
            .sum(['features', 'time']) > 0)
            .sum('tetrodes').values))
        pct_participating_tetrodes.append(
            100 * n_participating_tetrodes[-1] /
            len(data['multiunits'].tetrodes))
        median_pct_participating_tetrodes.append(
            100 * np.nanmedian((
                (~np.isnan(data['multiunits'].isel(
                    time=is_non_local_event)))
                .sum('features') > 0)
                .mean('tetrodes')))
        median_n_participating_tetrodes.append(
            np.nanmedian((
                (~np.isnan(data['multiunits'].isel(
                    time=is_non_local_event)))
                .sum('features') > 0)
                .sum('tetrodes')))

        swr_overlap_numbers = np.unique(
            data['ripple_labels'].iloc[is_non_local_event])
        swr_overlap_numbers = swr_overlap_numbers[
            swr_overlap_numbers > 0]

        overlap_swr_start_times = np.asarray(
            data['ripple_times'].loc[swr_overlap_numbers].start_time /
            np.timedelta64(1, 's'))
        overlap_swr_end_times = np.asarray(
            data['ripple_times'].loc[swr_overlap_numbers].end_time /
            np.timedelta64(1, 's'))

        mua_overlap_numbers = np.unique(
            data['multiunit_high_synchrony_labels']
            .iloc[is_non_local_event])
        mua_overlap_numbers = mua_overlap_numbers[
            mua_overlap_numbers > 0]

        overlap_mua_start_times = np.asarray(
            data['multiunit_high_synchrony_times'].loc[mua_overlap_numbers].start_time /
            np.timedelta64(1, 's'))
        overlap_mua_end_times = np.asarray(
            data['multiunit_high_synchrony_times'].loc[mua_overlap_numbers].end_time /
            np.timedelta64(1, 's'))

        non_local_event_start_time = np.asarray(
            results.time[is_non_local_event][0])
        non_local_event_end_time = np.asarray(
            results.time[is_non_local_event][-1])

        swr_start_time_offset.append(
            non_local_event_start_time - overlap_swr_start_times)
        swr_end_time_offset.append(
            non_local_event_end_time - overlap_swr_end_times)

        mua_start_time_offset.append(
            non_local_event_start_time - overlap_mua_start_times)
        mua_end_time_offset.append(
            non_local_event_end_time - overlap_mua_end_times)

    return pd.DataFrame(
        {"animal": animal,
         "day": day,
         "epoch": epoch,
         "event_number": event_number,
         "environment": environment,
         "duration": duration,
         "median_ripple_zscore": median_ripple_zscore,
         "median_multiunit_zscore": median_multiunit_zscore,
         "median_speed": median_speed,
         "median_hpd_pct": median_hpd_pct,
         "is_swr_overlap": is_swr_overlap,
         "is_mua_overlap": is_mua_overlap,
         "pct_overlap_swr": pct_overlap_swr,
         "pct_overlap_mua": pct_overlap_mua,
         "median_pct_distance_from_animal": median_pct_distance_from_animal,
         "median_replay_speed": median_replay_speed,
         "pct_participating_tetrodes": pct_participating_tetrodes,
         "n_participating_tetrodes": n_participating_tetrodes,
         "median_pct_participating_tetrodes": median_pct_participating_tetrodes,
         "median_n_participating_tetrodes": median_n_participating_tetrodes,
         "swr_start_time_offset": swr_start_time_offset,
         "swr_end_time_offset": swr_end_time_offset,
         "mua_start_time_offset": mua_start_time_offset,
         "mua_end_time_offset": mua_end_time_offset,
         "non_local_time_outside_swr": np.concatenate(
             non_local_time_outside_swr),
         "pct_non_local_time_outside_swr": np.concatenate(
             pct_non_local_time_outside_swr),
         "local_time_inside_swr": np.concatenate(local_time_inside_swr),
         "pct_local_time_inside_swr": np.concatenate(
             pct_local_time_inside_swr),
         "non_local_time_outside_mua": np.concatenate(
             non_local_time_outside_mua),
         "pct_non_local_time_outside_mua": np.concatenate(
             pct_non_local_time_outside_mua),
         "local_time_inside_mua": np.concatenate(local_time_inside_mua),
         "pct_local_time_inside_mua": np.concatenate(
             pct_local_time_inside_mua),
         }
    ).set_index(['animal', 'day', 'epoch', 'event_number'])
