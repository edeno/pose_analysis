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
from replay_identification import ReplayDetector
from src.load_data import load_data
from src.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


def setup_logging(epoch_key,
                  date_format='%d-%b-%y %H:%M:%S',
                  format='%(asctime)s %(message)s',
                  type=''):
    animal, day, epoch = epoch_key
    log_filename = (f"logs/{animal}_{day:02d}_{epoch:02d}"
                    f"_{type}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format, datefmt=date_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


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
    overwrite=True
):
    print(epoch_key)

    # setup_logging(epoch_key, type=f"{data_type}_non_local_{training_type}")

    epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
    results_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_identifier}_{data_type}_non_local_{training_type}.nc"
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

    detector_parameters = {
        'movement_var': 6.0,
        'replay_speed': 1,
        'place_bin_size': 2.0,
        'spike_model_knot_spacing': 12.0,
        'spike_model_penalty': 1E-5,
        'movement_state_transition_type': 'random_walk',
        'multiunit_model_kwargs': {
            'mark_std': 24.0,
            'position_std': 4.0,
            'block_size': 100},
        'discrete_state_transition_type': 'ripples_no_speed_threshold',
    }

    detector = ReplayDetector(**detector_parameters)
    logging.info(detector)

    if data_type == 'clusterless':

        # Garbage collect GPU memory
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        detector.fit(
            is_ripple=data['is_ripple'],
            is_training=is_training,
            speed=data['position_info'].nose_vel,
            position=data['position_info'].linear_position,
            multiunit=data['multiunits'],
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
            use_gpu=True,
        )

        results = detector.predict(
            speed=data['position_info'].nose_vel,
            multiunit=data['multiunits'],
            position=data['position_info'].linear_position,
            time=data['position_info'].index / np.timedelta64(1, 's'),
            use_likelihoods=['multiunit'],
            use_acausal=True,
            set_no_spike_to_equally_likely=False,
            use_gpu=False
        )
    elif data_type == 'sorted_spikes':
        detector.fit(
            is_ripple=data['is_ripple'],
            is_training=is_training,
            speed=data['position_info'].nose_vel,
            position=data['position_info'].linear_position,
            spikes=data['spikes'],
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
            use_gpu=False,
        )

        results = detector.predict(
            speed=data['position_info'].nose_vel,
            spikes=data['spikes'],
            position=data['position_info'].linear_position,
            time=data['position_info'].index / np.timedelta64(1, 's'),
            use_likelihoods=['spikes'],
            use_acausal=True,
            set_no_spike_to_equally_likely=False,
            use_gpu=False
        )
    else:
        logging.error('Data type not supported...')
        return

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

    return parser.parse_args()


def main():
    args = get_command_line_arguments()
    epoch_key = (args.Animal, args.Day, args.Epoch)

    decode(
        epoch_key,
        training_type=args.training_type,
        data_type=args.data_type,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    sys.exit(main())
