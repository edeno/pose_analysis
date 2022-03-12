import logging
import os
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import xarray as xr
from loren_frank_data_processing import make_epochs_dataframe
from replay_trajectory_classification import ClusterlessClassifier
from sklearn.model_selection import KFold
from src.load_data import load_data
from src.parameters import (ANIMALS, PROCESSED_DATA_DIR, WTRACK_EDGE_ORDER,
                            WTRACK_EDGE_SPACING)


def setup_logging(epoch_key,
                  date_format='%d-%b-%y %H:%M:%S',
                  format='%(asctime)s %(message)s'):
    animal, day, epoch = epoch_key
    log_filename = (f"logs/{animal}_{day:02d}_{epoch:02d}"
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


def run_analysis(epoch_key, overwrite=False):
    print(epoch_key)
    setup_logging(epoch_key)
    epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
    results_filename = os.path.join(
        PROCESSED_DATA_DIR,
        f"{epoch_identifier}_clusterless_forward_reverse_results.nc"
    )

    try:
        if not os.path.isfile(results_filename):
            logging.info(' START '.center(50, '#'))
            logging.info("Loading data...")
            data = load_data(epoch_key,
                             position_to_linearize=['nose_x', 'nose_y'],
                             max_distance_from_well=30,
                             min_distance_traveled=50,
                             )

            continuous_transition_types = [['random_walk_direction2', 'random_walk',            'uniform', 'random_walk',            'random_walk',            'uniform'],  # noqa
                                           ['random_walk',            'random_walk_direction1', 'uniform', 'random_walk',            'random_walk',            'uniform'],  # noqa
                                           ['uniform',                'uniform',                'uniform', 'uniform',                'uniform',                'uniform'],  # noqa
                                           ['random_walk',            'random_walk',            'uniform', 'random_walk_direction1', 'random_walk',            'uniform'],  # noqa
                                           ['random_walk',            'random_walk',            'uniform', 'random_walk',            'random_walk_direction2', 'uniform'],  # noqa
                                           ['uniform',                'uniform',                'uniform', 'uniform',                'uniform',                'uniform'],  # noqa
                                           ]
            encoding_group_to_state = ['Inbound', 'Inbound', 'Inbound',
                                       'Outbound', 'Outbound', 'Outbound']

            clusterless_algorithm = 'multiunit_likelihood_gpu'
            clusterless_algorithm_params = {
                'mark_std': 20.0,
                'position_std': 8.0,
            }

            classifier_parameters = {
                'movement_var': 6.0,
                'replay_speed': 1,
                'place_bin_size': 2.0,
                'continuous_transition_types': continuous_transition_types,
                'discrete_transition_diag': 0.968,
                'clusterless_algorithm': clusterless_algorithm,
                'clusterless_algorithm_params': clusterless_algorithm_params
            }

            logging.info(pprint(classifier_parameters))

            inbound_outbound_labels = np.asarray(
                data["position_info"].task).astype(str)

            notnull = pd.notnull(data["position_info"].task)

            state_names = [
                'Inbound-Forward', 'Inbound-Reverse', 'Inbound-Fragmented',
                'Outbound-Forward', 'Outbound-Reverse', 'Outbound-Fragmented']

            logging.info("Decoding...")
            classifier = ClusterlessClassifier(**classifier_parameters)

            logging.info("Fitting model...")
            classifier.fit(
                position=data["position_info"].linear_position,
                multiunits=data["multiunits"],
                is_training=notnull & (data['position_info'].nose_vel > 4),
                track_graph=data["track_graph"],
                edge_order=WTRACK_EDGE_ORDER,
                edge_spacing=WTRACK_EDGE_SPACING,
                encoding_group_labels=inbound_outbound_labels,
                encoding_group_to_state=encoding_group_to_state
            )
            # save the model
            classifier.save_model(
                os.path.join(
                    PROCESSED_DATA_DIR,
                    f"{epoch_identifier}_clusterless_forward_reverse_classifier.pkl"))
            logging.info('Predicting posterior...')
            results = classifier.predict(
                data["multiunits"],
                time=data["position_info"].index /
                np.timedelta64(1, "s"),
                state_names=state_names,
                use_gpu=True,
            )

            logging.info("Saving results...")
            results.to_netcdf(results_filename)
            logging.info('Done!\n')
        else:
            logging.info('File already processed. Skipping...\n')
    except Exception as e:
        logging.exception("Something bad happened")
        logging.shutdown()

    logging.shutdown()


def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    epoch_info = epoch_info.loc[(epoch_info.type == 'run') &
                                (epoch_info.environment == 'wtrack')]
    for epoch in epoch_info.index:
        run_analysis(epoch)


if __name__ == "__main__":
    main()
