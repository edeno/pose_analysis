from os.path import abspath, dirname, join, pardir

from loren_frank_data_processing import Animal

# LFP sampling frequency
SAMPLING_FREQUENCY = 500

# Data directories and definitions
ROOT_DIR = join(abspath(dirname(__file__)), pardir)
RAW_DATA_DIR = join(ROOT_DIR, 'Raw-Data')
PROCESSED_DATA_DIR = join(ROOT_DIR, 'Processed-Data')
FIGURE_DIR = join(ROOT_DIR, 'figures')

ANIMALS = {
    # 'jaq': Animal(directory=join(RAW_DATA_DIR, 'jaq'), short_name='jaq'),
    'Roqui': Animal(directory=join(RAW_DATA_DIR, 'Roqui'), short_name='Roqui'),
}

EDGE_ORDER = [0, 1, 3, 2, 4]
EDGE_SPACING = [15, 0, 15, 0]
