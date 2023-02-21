#!/usr/bin/env python
# coding: utf-8

"""Clusterless Decoding Analysis W-Track | Compute decoded position and
distance metrics based on CA1 Marks and associated body position |
Inputs: Marks, Posdlc, Task, Tetinfo | https://github.com/edeno/pose_analysis
"""
import logging
import os

import numpy as np
import xarray as xr
from loren_frank_data_processing import make_epochs_dataframe
from replay_trajectory_classification import ClusterlessClassifier
from sklearn.model_selection import KFold
from trajectory_analysis_tools import (
    get_distance_metrics,
    get_highest_posterior_threshold,
    get_HPD_spatial_coverage,
    get_trajectory_data,
)

from src.analysis import calculate_replay_distance
from src.load_data import load_data, make_track_graph
from src.parameters import (
    ANIMALS,
    EDGE_ORDER,
    EDGE_SPACING,
    PROCESSED_DATA_DIR,
    classifier_parameters,
    discrete_state_transition,
)

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="%d-%b-%y %H:%M:%S")


def run_analysis(epoch_key):
    logging.info(epoch_key)
    # Load Data
    # Specifiy animal, day, epoch and body position estimate to be used to
    # encode pos-mark relationship.
    logging.info("Loading data...")
    data = load_data(
        epoch_key,
        position_to_linearize=["nose_x", "nose_y"],
        max_distance_from_well=30,
        min_distance_traveled=50,
    )

    track_graph, center_well_id = make_track_graph(epoch_key, ANIMALS)
    is_running = np.abs(data["position_info"].nose_vel) > 0
    # is_running = np.abs(data["position_info"].forepawR_vel) > 4
    # is_outbound = data["position_info"].task == "Outbound"

    # Calculate posterior
    # Builds the classifier and calculates the posterior estimates for each
    # bin. Default is 5x cross validation. Some concerns if that is appropriate
    # in 15 minute run epochs,but AJ checked that the overal posterior was
    # similar in 2x,3x, and 5x versions. Maybe stick to 3x for 15 minute data?

    cv = KFold()
    cv_classifier_clusterless_results = []
    logging.info("Decoding...")
    for fold_ind, (train, test) in enumerate(cv.split(data["position_info"].index)):
        logging.info(f"\tFold #{fold_ind + 1}")
        # train = train[is_outbound[train].values]
        cv_classifier = ClusterlessClassifier(**classifier_parameters)
        logging.info("\tFitting model...")
        cv_classifier.fit(
            position=data["position_info"].iloc[train].linear_position,
            multiunits=data["multiunits"].isel(time=train),
            is_training=is_running.iloc[train],
            track_graph=track_graph,
            center_well_id=center_well_id,
            edge_order=EDGE_ORDER,
            edge_spacing=EDGE_SPACING,
        )
        cv_classifier.discrete_state_transition_ = discrete_state_transition
        logging.info("\tPredicting posterior...")
        cv_classifier_clusterless_results.append(
            cv_classifier.predict(
                data["multiunits"].isel(time=test),
                time=data["position_info"].iloc[test].index / np.timedelta64(1, "s"),
            )
        )
    # concatenate cv classifier results
    cv_classifier_clusterless_results = xr.concat(
        cv_classifier_clusterless_results, dim="time"
    )

    # Calculate Distance Metrics
    logging.info("Calculating metrics...")
    # Important calculate distance metrics. Loads Causal Posterior and
    # get_trajectory_data and get_distance_metrics to calculate ahead-behind
    # distance based on body_dir

    # CAUSAL
    posterior_causal = cv_classifier_clusterless_results["causal_posterior"].sum(
        "state", skipna=False
    )

    # extracting the peak of the posterior
    trajectory_data = get_trajectory_data(
        posterior=posterior_causal,
        track_graph=track_graph,
        decoder=cv_classifier,
        position_info=data["position_info"],
        direction_variable="body_dir",
    )

    distance_metrics = get_distance_metrics(track_graph, *trajectory_data)

    ahead_behind_distance_causal = (
        distance_metrics.mental_position_ahead_behind_animal
        * distance_metrics.mental_position_distance_from_animal
    )

    # Calculate the corresponding 95% HPD credible interval
    hpd_threshold_95_causal = get_highest_posterior_threshold(
        posterior_causal.dropna("position"), coverage=0.95
    )
    spatial_coverage_95_causal = get_HPD_spatial_coverage(
        posterior_causal, hpd_threshold_95_causal
    )

    # Calculate the corresponding 50% HPD credible interval
    hpd_threshold_50_causal = get_highest_posterior_threshold(
        posterior_causal.dropna("position"), coverage=0.50
    )
    spatial_coverage_50_causal = get_HPD_spatial_coverage(
        posterior_causal, hpd_threshold_50_causal
    )

    # calculate distance metrics acausal posterior. Loads acausal posterior and
    # distance metrics. ACAUSAL
    posterior_acausal = cv_classifier_clusterless_results["acausal_posterior"].sum(
        "state", skipna=False
    )

    # extracting the peak of the posterior
    trajectory_data = get_trajectory_data(
        posterior=posterior_acausal,
        track_graph=track_graph,
        decoder=cv_classifier,
        position_info=data["position_info"],
        direction_variable="body_dir",
    )

    distance_metrics = get_distance_metrics(track_graph, *trajectory_data)

    ahead_behind_distance_acausal = (
        distance_metrics.mental_position_ahead_behind_animal
        * distance_metrics.mental_position_distance_from_animal
    )

    # ACAUSAL 95% CI
    hpd_threshold_95_acausal = get_highest_posterior_threshold(
        posterior_acausal.dropna("position"), coverage=0.95
    )
    spatial_coverage_95_acausal = get_HPD_spatial_coverage(
        posterior_acausal, hpd_threshold_95_acausal
    )

    # ACAUSAL 50% CI

    hpd_threshold_50_acausal = get_highest_posterior_threshold(
        posterior_acausal.dropna("position"), coverage=0.50
    )
    spatial_coverage_50_acausal = get_HPD_spatial_coverage(
        posterior_acausal, hpd_threshold_50_acausal
    )

    # WHILE WE ARE AT IT, ALSO A GOOD IDEA TO CALCULATE THE ABSOLUTE DISTANCE.
    # CAUSAL
    replay_distance_from_animal_position_causal = calculate_replay_distance(
        posterior=cv_classifier_clusterless_results.causal_posterior.sum("state"),
        track_graph=track_graph,
        decoder=cv_classifier,
        position_2D=data["position_info"].loc[:, ["nose_x", "nose_y"]],
        track_segment_id=data["position_info"].loc[:, ["track_segment_id"]],
    )

    # WHILE WE ARE AT IT, ALSO A GOOD IDEA TO CALCULATE THE ABSOLUTE DISTANCE.
    # ACAUSAL
    replay_distance_from_animal_position_acausal = calculate_replay_distance(
        posterior=cv_classifier_clusterless_results.acausal_posterior.sum("state"),
        track_graph=track_graph,
        decoder=cv_classifier,
        position_2D=data["position_info"].loc[:, ["nose_x", "nose_y"]],
        track_segment_id=data["position_info"].loc[:, ["track_segment_id"]],
    )

    # ### Save the distance and CI values with the classifier results
    cv_classifier_clusterless_results["abs_distance_from_animal_position_causal"] = (
        ("time"),
        replay_distance_from_animal_position_causal,
    )
    cv_classifier_clusterless_results["abs_distance_from_animal_position_acausal"] = (
        ("time"),
        replay_distance_from_animal_position_acausal,
    )

    # maybe this will works and we can save both distances
    cv_classifier_clusterless_results["rel_distance_from_animal_position_causal"] = (
        ("time"),
        ahead_behind_distance_causal,
    )
    cv_classifier_clusterless_results["rel_distance_from_animal_position_acausal"] = (
        ("time"),
        ahead_behind_distance_acausal,
    )

    # get HPD estimate of the distance associated
    cv_classifier_clusterless_results["hpd_threshold_95_causal"] = (
        ("time"),
        hpd_threshold_95_causal,
    )
    cv_classifier_clusterless_results["hpd_threshold_50_causal"] = (
        ("time"),
        hpd_threshold_50_causal,
    )
    cv_classifier_clusterless_results["hpd_threshold_95_acausal"] = (
        ("time"),
        hpd_threshold_95_acausal,
    )
    cv_classifier_clusterless_results["hpd_threshold_50_acausal"] = (
        ("time"),
        hpd_threshold_50_acausal,
    )

    # get CI of the distance associated
    cv_classifier_clusterless_results["credible_interval_95_causal"] = (
        ("time"),
        spatial_coverage_95_causal,
    )
    cv_classifier_clusterless_results["credible_interval_50_causal"] = (
        ("time"),
        spatial_coverage_50_causal,
    )
    cv_classifier_clusterless_results["credible_interval_95_acausal"] = (
        ("time"),
        spatial_coverage_95_acausal,
    )
    cv_classifier_clusterless_results["credible_interval_50_acausal"] = (
        ("time"),
        spatial_coverage_50_acausal,
    )

    logging.info("Saving results...")
    # save the results as .nc format. ncread matlab can read these
    epoch_identifier = f"{epoch_key[0]}_{epoch_key[1]:02d}_{epoch_key[2]:02d}"
    cv_classifier_clusterless_results.to_netcdf(
        os.path.join(
            PROCESSED_DATA_DIR,
            (
                f"{epoch_identifier}_cv_classifier_clusterless_vel_0_nose_alltime"
                "_results.nc"
            ),
        )
    )

    # save the model
    cv_classifier.save_model(
        os.path.join(
            PROCESSED_DATA_DIR, f"{epoch_identifier}_cv_classifier_clusterless_nose.pkl"
        )
    )

    # Save position info
    data["position_info"].to_csv(
        os.path.join(PROCESSED_DATA_DIR, f"{epoch_identifier }_position_info_nose.csv")
    )


def main():
    epoch_info = make_epochs_dataframe(ANIMALS)
    for epoch in epoch_info.index:
        run_analysis(epoch)


if __name__ == "__main__":
    main()
