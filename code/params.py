# Import necessary modules
import numpy as np

# Directories
DATA_DIR = '../data/from_osf'
EEG_DIR = 'EEG'

# Experimental parameters
EXP = 1
SUBJ = 1
N_BLOCKS = 3
N_BLOCK_ITERS = 10
ALPHA_BAND = (8, 12)  # Hz

# IEM parameters
IEM_N_CHANNELS = 8
IEM_FEAT_SPACE_EDGES = (0, 359)
IEM_BASIS_FUNC = lambda theta: np.sin(0.5 * np.radians(theta)) ** 7 
FEATURE_NAME = 'Spatial Location'