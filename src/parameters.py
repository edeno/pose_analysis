from os.path import abspath, dirname, join, pardir

import numpy as np
from loren_frank_data_processing import Animal
import socket

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, "Raw-Data")
PROCESSED_DATA_DIR = join(ROOT_DIR, "Processed-Data")
FIGURE_DIR = join(ROOT_DIR, "figures")


hostname = socket.gethostname()

if hostname[-12:] == "cin.ucsf.edu":
    ANIMALS = {
        "Jaq": Animal(
            directory="/stelmo/abhilasha/animals/Jaq/filterframework", short_name="Jaq"
        ),
        "Roqui": Animal(
            directory="/stelmo/abhilasha/animals/Roqui/filterframework",
            short_name="Roqui",
        ),
        "Peanut": Animal(
            directory="/stelmo/abhilasha/animals/Peanut/filterframework",
            short_name="Peanut",
        ),
        "Lotus": Animal(
            directory="/stelmo/abhilasha/animals/Lotus/filterframework",
            short_name="Lotus",
        ),
        "Monty": Animal(
            directory="/stelmo/abhilasha/animals/Monty/filterframework",
            short_name="Monty",
        ),
    }
else:
    ANIMALS = {
        "Jaq": Animal(directory=join(RAW_DATA_DIR, "Jaq")),
        "Roqui": Animal(directory=join(RAW_DATA_DIR, "Roqui")),
        "Peanut": Animal(directory=join(RAW_DATA_DIR, "Peanut")),
        "Lotus": Animal(directory=join(RAW_DATA_DIR, "Lotus")),
        "Monty": Animal(directory=join(RAW_DATA_DIR, "Monty")),
    }

WTRACK_EDGE_ORDER = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]
WTRACK_EDGE_SPACING = [15, 0, 15, 0]

LINEAR_EDGE_ORDER = [(0, 1)]
LINEAR_EDGE_SPACING = 0

classifier_parameters = {
    "movement_var": 6.0,
    "replay_speed": 1,
    "place_bin_size": 2.5,
    "continuous_transition_types": [["random_walk", "uniform"], ["uniform", "uniform"]],
    "model_kwargs": {
        "bandwidth": np.array([20.0, 20.0, 20.0, 20.0, 8.0])
    },  # 1-4 values correspond to the variance of the four mark dimensions and the 5th value is the variance of the position dimension
}

discrete_state_transition = np.array([[0.968, 0.032], [0.032, 0.968]])

"""
1. Geometric mean of duration is 1 / (1 - p)
2. So p = 1 - (1 / n_time_steps).
3. Want `n_time_steps` to equal half a theta cycle.
4. Theta cycles are ~8 Hz or 125 ms per cycle.
5. Half a theta cycle is 62.5 ms.
6. If our timestep is 2 ms, then n_time_steps = 31.25
7. So p = 0.968
"""
