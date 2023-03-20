"""All parameters for decoding spatial working memory (WM) from EEG data."""

# Import necessary modules
import numpy as np

# Directories
DATA_DIR = '/home/AD/abender/Documents/decoding_spatial_wm/data'
DOWNLOAD_DIR = f'{DATA_DIR}/by_experiment'
PROCESSED_DIR = f'{DATA_DIR}/by_subject'
TOTAL_POWER_DIR = f'{DATA_DIR}/total_power'
TFR_DIR = f'{DATA_DIR}/tfr'
SPARAM_DIR = f'{DATA_DIR}/sparam'
IEM_OUTPUT_DIR = f'{DATA_DIR}/iem_output'
EEG_DIR = 'EEG'
FIG_DIR = 'figs'

# Experiment variable
NUM_SUBJECTS = {'CB1': 12, 'CB2': 32, 'CS': 24, 'JNP': 44, 'SB': 36}
BAD_SUBJECTS = {'CB1': (),
                'CB2': (25, 27),
                'CS':  (0, 4, 6, 18),
                'JNP': (32,),
                'SB':  ()}
EXPERIMENT_VARS = {
    'CB1': {'data': ('data', 'eeg', 'arf', 'trial', 'data'),
            'ch_labels': ('data', 'eeg', 'arf', 'chanLabels'),
            'sfreq': 250,
            'pre_time': ('data', 'eeg', 'preTime'),
            'post_time': ('data', 'eeg', 'postTime'),
            'bad_trials': ('data', 'eeg', 'arf', 'artifactIndCleaned'),
            'pos_bin': ('data', 'beh', 'posBinRot'),
            'bad_electrodes': None,
            'art_pre_time': ('data', 'eeg', 'artPreTime'),
            'art_post_time': ('data', 'eeg', 'artPostTIme')},
    'CB2': {'data': ('data', 'eeg', 'trial', 'data'),
            'ch_labels': ('data', 'eeg', 'chanLabels'),
            'sfreq': 500,
            'pre_time': ('data', 'eeg', 'preTime'),
            'post_time': ('data', 'eeg', 'postTime'),
            'bad_trials': ('data', 'eeg', 'arf', 'artifactIndCleaned'),
            'pos_bin': ('data', 'beh', 't', 'posBinRot'),
            'bad_electrodes': ('data', 'eeg', 'droppedElectrodes'),
            'art_pre_time': ('data', 'eeg', 'arfPreTime'),
            'art_post_time': ('data', 'eeg', 'arfPostTime')},
    'CS':  {'data': ('data', 'eeg', 'trial', 'baselined'),
            'ch_labels': ('data', 'eeg', 'chanLabels'),
            'sfreq': ('data', 'eeg', 'srate'),
            'pre_time': ('data', 'eeg', 'settings', 'seg', 'preTime'),
            'post_time': ('data', 'eeg', 'settings', 'seg', 'postTime'),
            'bad_trials': ('data', 'eeg', 'arf', 'artifactIndCleaned'),
            'pos_bin': ('data', 'beh', 'posBinRot'),
            'bad_electrodes': ('data', 'eeg', 'droppedElectrodes'),
            'art_pre_time': ('data', 'eeg', 'settings', 'seg', 'arfPreTime'),
            'art_post_time': ('data', 'eeg', 'settings', 'seg', 'arfPostTime')},
    'JNP': {'data': ('data', 'eeg', 'data'),
            'ch_labels': ('data', 'eeg', 'chanLabels'),
            'sfreq': ('data', 'eeg', 'sampRate'),
            'pre_time': ('data', 'eeg', 'preTime'),
            'post_time': ('data', 'eeg', 'postTime'),
            'bad_trials': ('data', 'eeg', 'artIndCleaned'),
            'pos_bin': ('data', 'beh', 'posBinRot'),
            'bad_electrodes': None,
            'art_pre_time': None,
            'art_post_time': None},
    'SB':  {'data': ('data', 'eeg', 'trial', 'data'),
            'ch_labels': ('data', 'eeg', 'chanLabels'),
            'sfreq': ('data', 'eeg', 'srate'),
            'pre_time': ('data', 'eeg', 'preTime'),
            'post_time': ('data', 'eeg', 'postTime'),
            'bad_trials': ('data', 'eeg', 'arf', 'artifactIndCleaned'),
            'pos_bin': ('data', 'beh', 'stim', 'posBin2tf'),
            'bad_electrodes': ('data', 'eeg', 'droppedElectrodes'),
            'art_pre_time': ('data', 'eeg', 'arfPreTime'),
            'art_post_time': ('data', 'eeg', 'arfPostTime')}}

# Computing parameters
N_CPUS = 12
NICENESS = 20

# Experimental parameters
EXPERIMENT_NUM = 1
SUBJ = 1
N_BLOCKS = 3
N_BLOCK_ITERS = 10
ALPHA_BAND = (8, 12)  # Hz

# Spectral estimation parameters
FMIN = 2  # Hz
FMAX = 50  # Hz
N_FREQS = 128
TIME_WINDOW_LEN = 0.5  # s
DECIM_FACTOR = 4  # decimation/downsampling factor
N_PEAKS = 4
PEAK_WIDTH_LIMS = (2, 8)

# IEM parameters
IEM_N_CHANNELS = 8
IEM_FEAT_SPACE_EDGES = (0, 359)
IEM_BASIS_FUNC = lambda theta: np.sin(0.5 * np.radians(theta)) ** 7
FEATURE_NAME = 'Spatial Location'
