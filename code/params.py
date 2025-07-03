"""All parameters for decoding spatial working memory (WM) from EEG data."""

# Import necessary modules
import numpy as np

# Directories
DATA_DIR = "/labs/bvoyteklab/Bender/decoding_spatial_wm/data"
DOWNLOAD_DIR = f"{DATA_DIR}/by_experiment"
EPOCHS_DIR = f"{DATA_DIR}/by_subject"
TOTAL_POWER_DIR = f"{DATA_DIR}/total_power"
SPARAM_DIR = f"{DATA_DIR}/sparam"
ERP_DIR = f"{DATA_DIR}/erp"
IEM_OUTPUT_DIR = f"{DATA_DIR}/iem_output"
CRR_OUTPUT_DIR = f"{DATA_DIR}/crr_output"
SINGLE_TRIAL_IEM_DIR = f"{DATA_DIR}/single_trial_iem"
SINGLE_TRIAL_CRR_DIR = f"{DATA_DIR}/single_trial_crr"
FIG_DIR = "figs"

# Experiment variable
SUBJECTS_BY_TASK = (
    ("JNP", tuple(range(15))),
    ("JNP", tuple(range(15, 30))),
    ("JNP", tuple(range(30, 44))),
    ("CB1", tuple(range(12))),
    ("CB2", tuple(range(18))),
    ("CB2", tuple(range(18, 32))),
    ("CS", tuple(range(24))),
)
#     ('SB', tuple(range(36))))
TASK_TIMINGS = (
    (0.25, 2.0),
    (1.0, 2.0),
    (0.25, 2.0),
    (0.1, 1.3),
    (0.1, 1.25),
    (0.1, 1.25),
    (0.25, 1.25),
    (0.25, 1.3),
)
NUM_SUBJECTS = {"CB1": 12, "CB2": 32, "CS": 24, "JNP": 44, "SB": 36}
BAD_SUBJECTS = {
    "CB1": (),
    "CB2": (25, 27),
    "CS": (0, 4, 6, 18),
    "JNP": (32,),
    "SB": (),
}
EXPERIMENT_VARS = {
    "CB1": {
        "data": ("data", "eeg", "arf", "trial", "data"),
        "ch_labels": ("data", "eeg", "arf", "chanLabels"),
        "sfreq": 250,
        "pre_time": ("data", "eeg", "preTime"),
        "post_time": ("data", "eeg", "postTime"),
        "bad_trials": ("data", "eeg", "arf", "artifactIndCleaned"),
        "pos_bin": ("data", "beh", "posBinRot"),
        "pos": ("data", "beh", "pos"),
        "pos_backup": None,
        "pos_bin_nt": None,
        "pos_nt": None,
        "bad_electrodes": None,
        "art_pre_time": ("data", "eeg", "artPreTime"),
        "art_post_time": ("data", "eeg", "artPostTIme"),
        "error": ("data", "beh", "err"),
        "rt": ("data", "beh", "rt"),
    },
    "CB2": {
        "data": ("data", "eeg", "trial", "data"),
        "ch_labels": ("data", "eeg", "chanLabels"),
        "sfreq": 500,
        "pre_time": ("data", "eeg", "preTime"),
        "post_time": ("data", "eeg", "postTime"),
        "bad_trials": ("data", "eeg", "arf", "artifactIndCleaned"),
        "pos_bin": ("data", "beh", "t", "posBinRot"),
        "pos": ("data", "beh", "t", "pos"),
        "pos_backup": None,
        "pos_bin_nt": ("data", "beh", "nt", "posBin"),
        "pos_nt": ("data", "beh", "nt", "pos"),
        "bad_electrodes": ("data", "eeg", "droppedElectrodes"),
        "art_pre_time": ("data", "eeg", "arfPreTime"),
        "art_post_time": ("data", "eeg", "arfPostTime"),
        "error": ("data", "beh", "colOffset"),
        "rt": ("data", "beh", "rt"),
    },
    "CS": {
        "data": ("data", "eeg", "trial", "baselined"),
        "ch_labels": ("data", "eeg", "chanLabels"),
        "sfreq": ("data", "eeg", "srate"),
        "pre_time": ("data", "eeg", "settings", "seg", "preTime"),
        "post_time": ("data", "eeg", "settings", "seg", "postTime"),
        "bad_trials": ("data", "eeg", "arf", "artifactIndCleaned"),
        "pos_bin": ("data", "beh", "posBinRot"),
        "pos": ("data", "beh", "probedPos"),
        "pos_backup": None,
        "pos_bin_nt": ("data", "beh", "unprobedPosBin"),
        "pos_nt": ("data", "beh", "unprobedPos"),
        "bad_electrodes": ("data", "eeg", "droppedElectrodes"),
        "art_pre_time": ("data", "eeg", "settings", "seg", "arfPreTime"),
        "art_post_time": ("data", "eeg", "settings", "seg", "arfPostTime"),
        "error": ("data", "beh", "angleOffset"),
        "rt": ("data", "beh", "rt"),
    },
    "JNP": {
        "data": ("data", "eeg", "data"),
        "ch_labels": ("data", "eeg", "chanLabels"),
        "sfreq": ("data", "eeg", "sampRate"),
        "pre_time": ("data", "eeg", "preTime"),
        "post_time": ("data", "eeg", "postTime"),
        "bad_trials": ("data", "eeg", "artIndCleaned"),
        "pos_bin": ("data", "beh", "posBinRot"),
        "pos": ("data", "beh", "pos2"),
        "pos_backup": ("data", "beh", "pos"),
        "pos_bin_nt": None,
        "pos_nt": None,
        "bad_electrodes": None,
        "art_pre_time": None,
        "art_post_time": None,
        "error": ("data", "beh", "err"),
        "rt": None,
    },
}

# Computing parameters
NICENESS = 19

# Experimental parameters
EXPERIMENT_NUM = 1
SUBJ = 1
N_BLOCKS = 3
N_BLOCK_ITERS = 100
ALPHA_BAND = (8, 12)  # Hz

# Spectral estimation parameters
FMIN = 2  # Hz
FMAX = 50  # Hz
N_FREQS = 128
TIME_WINDOW_LEN = 1.0  # s
DECIM_FACTOR = 1  # decimation/downsampling factor
N_PEAKS = 4
PEAK_WIDTH_LIMS = (2, 8)

# IEM/CRR parameters
SEED = 7
IEM_N_CHANNELS = 8
IEM_FEAT_SPACE_EDGES = (0, 359)
IEM_BASIS_FUNC = lambda theta: np.sin(0.5 * np.radians(theta)) ** 7
FEATURE_NAME = "Spatial Location"

# Electrode positions
JNP_MONTAGE = "standard_1020"
JNP_CUSTOM_CHANNELS = {}

# Plotting parameters
CHANNELS_TO_PLOT = ["O1", "O2", "Fz", "Pz"]
PLOT_FREQ_RANGE = (1, 30)
PLOT_SETTINGS = {
    # "font.family": "Helvetica",
    # "font.sans-serif": "Helvetica",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
PARAMS_TO_PLOT = {
    "total_power": {
        "dir": TOTAL_POWER_DIR,
        "name": "Alpha total power",
        "color": "#489ded",
    },
    "linOscAUC": {
        "dir": SPARAM_DIR,
        "name": "Alpha oscillatory power",
        "color": "#7a65b9",
    },
    "exponent": {
        "dir": SPARAM_DIR,
        "name": "Aperiodic exponent",
        "color": "#2e2e2e",
    },
}
