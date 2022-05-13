"""All parameters for decoding spatial working memory (WM) from EEG data."""

# Import necessary modules
import os
import numpy as np

# Directories
DATA_DIR = 'data'
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'from_osf')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CHANNEL_OFFSETS_DIR = os.path.join(DATA_DIR, 'channel_offsets')
EEG_DIR = 'EEG'
FIG_DIR = 'figs'

# Experimental parameters
EXPERIMENT_NUM = 1
SUBJ = 1
N_BLOCKS = 3
N_BLOCK_ITERS = 10
ALPHA_BAND = (8, 12)  # Hz

# Spectral estimation parameters
FMIN = 2  # Hz
FMAX = 100  # Hz
N_FREQS = 128
TIME_WINDOW_LEN = 0.5  # s
DECIM_FACT = 4  # decimation/downsampling factor
N_PEAKS = 4
N_CPUS = 6
SPARAM_VARS = (
    ('aperiodic_params', 'exponent', 'exp'),
    ('aperiodic_params', 'offset', 'offset'),
    ('peak_params', 'CF', 'freq'),
    ('peak_params', 'PW', 'power'),
    ('peak_params', 'BW', 'width'))  # (data field, var/column, desired name)

# IEM parameters
IEM_N_CHANNELS = 8
IEM_FEAT_SPACE_EDGES = (0, 359)
IEM_BASIS_FUNC = lambda theta: np.sin(0.5 * np.radians(theta)) ** 7
FEATURE_NAME = 'Spatial Location'
