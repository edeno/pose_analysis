#!/usr/bin/env python
# coding: utf-8

import logging
import pprint
import sys

import numba
import numpy as np
from replay_trajectory_classification import ClusterlessClassifier
from src.load_data import load_data
from src.parameters import WTRACK_EDGE_ORDER, WTRACK_EDGE_SPACING


def setup_logging(epoch_key,
                  date_format='%d-%b-%y %H:%M:%S',
                  format='%(asctime)s %(message)s'):
    animal, day, epoch = epoch_key
    log_filename = (f"{animal}_{day: 02d}_{epoch: 02d}"
                    "_clusterless_forward_reverse.log")

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


def main():
    epoch_key = 'Jaq', 3, 12

    logging.info('GPU likelihood + State Space')
    data = load_data(epoch_key,
                     position_to_linearize=['nose_x', 'nose_y'],
                     max_distance_from_well=5,
                     min_distance_traveled=30)

    continuous_transition_types = (
        [['random_walk', 'uniform'],
         ['uniform',     'uniform']])

    clusterless_algorithm = 'multiunit_likelihood_gpu'
    clusterless_algorithm_params = {
        'mark_std': 20.0,
        'position_std': 8.0,
    }

    classifier_parameters = {
        'movement_var': 6.0,
        'replay_speed': 1,
        'place_bin_size': 2.5,
        'continuous_transition_types': continuous_transition_types,
        'discrete_transition_diag': 0.968,
        'clusterless_algorithm': clusterless_algorithm,
        'clusterless_algorithm_params': clusterless_algorithm_params
    }

    logging.info(pprint.pprint(classifier_parameters))

    state_names = ['Continuous', 'Fragmented']

    classifier = ClusterlessClassifier(**classifier_parameters)
    classifier.fit(
        position=data["position_info"].linear_position,
        multiunits=data["multiunits"],
        track_graph=data["track_graph"],
        edge_order=WTRACK_EDGE_ORDER,
        edge_spacing=WTRACK_EDGE_SPACING,
    )
    numba.cuda.profile_start()
    results_gpu_likelihood = classifier.predict(
        data["multiunits"],
        time=data["position_info"].index / np.timedelta64(1, "s"),
        state_names=state_names,
        use_gpu=True
    )
    numba.cuda.profile_stop()
    logging.info('Done...')


if __name__ == "__main__":
    main()
