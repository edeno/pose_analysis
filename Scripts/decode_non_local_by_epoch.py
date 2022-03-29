import logging
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import numpy as np
from replay_identification.detectors import (ClusterlessDetector,
                                             SortedSpikesDetector)
from src.load_data import load_data
from src.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


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


def decode(
    epoch_key,
    training_type='no_ripple_and_no_ascending_theta',
    data_type='clusterless',
    use_EM=True,
    overwrite=True,
):
    print(epoch_key)

    epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
    if use_EM:
        em_str = '_EM'
    else:
        em_str = ''
    results_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_identifier}_{data_type}_non_local_{training_type}{em_str}.nc"
    )

    logging.info(' START '.center(50, '#'))

    git_hash = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        stdout=subprocess.PIPE, universal_newlines=True).stdout
    logging.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

    logging.info(f'data type: {data_type}')
    logging.info(f'training type: {training_type}')

    if Path(results_filename).is_file() and not overwrite:
        logging.info("Found existing results and not overwriting...")
        logging.info('Done!\n')
        return

    logging.info("Loading data...")
    data = load_data(epoch_key,
                     position_to_linearize=['nose_x', 'nose_y'],
                     max_distance_from_well=30,
                     min_distance_traveled=50,
                     )

    # estimate by ripple
    is_training = get_is_training(data, training_type=training_type)

    if data_type == 'clusterless':

        # Garbage collect GPU memory
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        detector_parameters = dict(
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
            clusterless_algorithm='multiunit_likelihood_integer_gpu',
            clusterless_algorithm_params=dict(
                mark_std=24.0, position_std=6.0, block_size=100),
        )

        detector = ClusterlessDetector(**detector_parameters)
        logging.info(detector)

        fit_args = dict(
            is_training=is_training,
            position=data['position_info'].linear_position,
            multiunits=data['multiunits'],
        )

        predict_args = dict(
            position=data['position_info'].linear_position,
            multiunits=data['multiunits'],
            time=data['position_info'].index / np.timedelta64(1, 's'),
        )

    elif data_type == 'sorted_spikes':

        detector_parameters = dict(
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
            spike_model_knot_spacing=12.5,
        )
        detector = SortedSpikesDetector(**detector_parameters)

        fit_args = dict(
            is_training=is_training,
            position=data['position_info'].linear_position,
            spikes=data['spikes'],
        )

        predict_args = dict(
            position=data['position_info'].linear_position,
            spikes=data['spikes'],
            time=data['position_info'].index / np.timedelta64(1, 's'),
        )

    else:
        logging.error('Data type not supported...')
        return

    if use_EM:
        results, data_log_likelihood = detector.estimate_parameters(
            fit_args,
            predict_args,
            estimate_state_transition=True,
            estimate_likelihood=True,
            max_iter=20,
        )
        results.attrs['data_log_likelihoods'] = data_log_likelihood
    else:
        detector.fit(**fit_args)
        results = detector.predict(**predict_args)

    results.assign(discrete_state_transition=(
        ['state', 'state'], detector.discrete_state_transition_))

    logging.info("Saving results...")
    results.to_netcdf(results_filename)
    logging.info('Done!\n')


def get_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument('Animal', type=str, help='Short name of animal')
    parser.add_argument('Day', type=int, help='Day of recording session')
    parser.add_argument('Epoch', type=int,
                        help='Epoch number of recording session')
    parser.add_argument('--data_type', type=str, default='clusterless')
    parser.add_argument('--training_type', type=str,
                        default='no_ripple_and_no_ascending_theta')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--use_EM', action='store_true')

    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    epoch_key = (args.Animal, args.Day, args.Epoch)

    decode(
        epoch_key,
        training_type=args.training_type,
        data_type=args.data_type,
        overwrite=args.overwrite,
        use_EM=args.use_EM,
    )


if __name__ == '__main__':
    sys.exit(main())
