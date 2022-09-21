import logging
import os
import subprocess
from pathlib import Path

from loren_frank_data_processing import make_epochs_dataframe
from replay_trajectory_classification import (ClusterlessClassifier,
                                              SortedSpikesClassifier)
from src.parameters import ANIMALS

try:
    import cupy as cp
except ImportError:
    import numpy as cp

import dask
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from replay_trajectory_classification.continuous_state_transitions import (
    RandomWalk, Uniform)
from replay_trajectory_classification.environments import Environment
from src.load_data import load_data
from src.parameters import PROCESSED_DATA_DIR


def setup_logger(name_logfile, path_logfile):
    """Sets up a logger for each function that outputs
    to the console and to a file"""
    logger = logging.getLogger(name_logfile)
    formatter = logging.Formatter(
        '%(asctime)s %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    return logger


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


@dask.delayed
def decode(
    epoch_key,
    log_directory='',
    training_type='no_ripple',
    data_type='clusterless',
    overwrite=True,
):
    try:
        epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"

        # Create a log file
        file_name = f"{epoch_identifier}_{data_type}_contfrag_{training_type}"
        logger = setup_logger(
            name_logfile=file_name,
            path_logfile=os.path.join(
                log_directory, f'{file_name}.log'))

        logger.info(' START '.center(50, '#'))

        # Check if results exist
        results_filename = os.path.join(
            PROCESSED_DATA_DIR, f"{file_name}.nc"
        )
        if Path(results_filename).is_file() and not overwrite:
            logger.info("Found existing results and not overwriting...")
            logger.info('Done!\n')
            return

        git_hash = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            stdout=subprocess.PIPE, universal_newlines=True).stdout
        logger.info('Git Hash: {git_hash}'.format(git_hash=git_hash.rstrip()))

        logger.info(f'data type: {data_type}')
        logger.info(f'training type: {training_type}')

        logger.info("Loading data...")
        data = load_data(epoch_key,
                         position_to_linearize=['nose_x', 'nose_y'],
                         max_distance_from_well=30,
                         min_distance_traveled=50,
                         )

        # estimate by ripple
        is_training = get_is_training(data, training_type=training_type)

        environment = Environment(
            place_bin_size=2.0,
            track_graph=data['track_graph'],
            edge_order=data['edge_order'],
            edge_spacing=data['edge_spacing'],
        )
        continuous_transition_types = [
            [RandomWalk(movement_var=6.0),  Uniform()],
            [Uniform(),                     Uniform()]]
        state_names = ['Continuous', 'Fragmented']

        if data_type == 'clusterless':

            # Garbage collect GPU memory
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

            classifier_parameters = dict(
                environments=environment,
                continuous_transition_types=continuous_transition_types,
                clusterless_algorithm='multiunit_likelihood_integer_gpu',
                clusterless_algorithm_params=dict(
                    mark_std=24.0, position_std=6.0, block_size=4096),
            )

            classifier = ClusterlessClassifier(**classifier_parameters)

            fit_args = dict(
                is_training=is_training,
                position=data['position_info'].linear_position,
                multiunits=data['multiunits'],
            )

            predict_args = dict(
                multiunits=data['multiunits'],
                time=data['position_info'].index / np.timedelta64(1, 's'),
                state_names=state_names,
                use_gpu=True,
            )

        elif data_type == 'sorted_spikes':

            classifier_parameters = dict(
                environments=environment,
                continuous_transition_types=continuous_transition_types,
                sorted_spikes_algorithm='spiking_likelihood_glm',
                sorted_spikes_algorithm_params={
                    'spike_model_knot_spacing': 12.5,
                }
            )
            classifier = SortedSpikesClassifier(**classifier_parameters)

            fit_args = dict(
                is_training=is_training,
                position=data['position_info'].linear_position,
                spikes=data['spikes'],
            )

            predict_args = dict(
                spikes=data['spikes'],
                time=data['position_info'].index / np.timedelta64(1, 's'),
                state_names=state_names,
                use_gpu=True,
            )

        else:
            logger.error('Data type not supported...')
            return

        logger.info(classifier)
        logger.info("Fitting model...")
        classifier.fit(**fit_args)
        logger.info("Predicting posterior...")
        results = classifier.predict(**predict_args)

        logger.info("Saving results...")
        results.to_netcdf(results_filename)
        logger.info('Done!\n')
    except Exception as e:
        logger.warning(e)


if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)

    log_directory = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_directory,  exist_ok=True)

    epoch_info = make_epochs_dataframe(ANIMALS)
    epoch_info = epoch_info.loc[(epoch_info.type == 'run')]
    # Append the result of the computation into a results list
    results = [decode(epoch_key, log_directory)
               for epoch_key in epoch_info.index]

    # Run `dask.compute` on the results list for the code to run
    dask.compute(*results)
