"""All parameters for decoding spatial working memory (WM) from EEG data."""

# Import necessary modules
import os
import numpy as np

# Directories
DATA_DIR = 'data'
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'from_osf')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
EEG_DIR = 'EEG'

# Experimental parameters
EXPERIMENT_NUM = 1
SUBJ = 1
N_BLOCKS = 3
N_BLOCK_ITERS = 10
ALPHA_BAND = (8, 12)  # Hz

# IEM parameters
IEM_N_CHANNELS = 8
IEM_FEAT_SPACE_EDGES = (0, 359)
IEM_BASIS_FUNC = lambda theta: np.sin(0.5 * np.radians(theta)) ** 7
FEATURE_NAME = 'Spatial Location'
