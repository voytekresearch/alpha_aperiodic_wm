"""Load EEG and behavioral data from MAT files downloaded from OSF
(https://osf.io/bwzfj/)"""

# Import necessary modules
import os 
import mne
from mne.externals.pymatreader import read_mat
from params import *


def load_eeg_one_subj(
    subj, data_dir=DATA_DIR, eeg_dir=EEG_DIR, exp_num=EXP_NUM):
    """Load EEG data for one subject."""
    # Load data from MAT file
    eeg_mat_fname = os.path.join(data_dir, 'exp{}'.format(
        exp_num), eeg_dir, '{}_EEG.mat'.format(subj))
    eeg_data = read_mat(eeg_mat_fname)['eeg']

    # Create epochs array from loaded MAT file
    info = mne.create_info(
        eeg_data['chanLabels'], eeg_data['sampRate'], ch_types='eeg')
    epochs = mne.EpochsArray(
        eeg_data['data'], info, tmin=eeg_data['preTime'] / 1000).drop(
            eeg_data['arf']['artIndCleaned'].astype(bool))
    assert epochs.times[-1] == eeg_data['postTime'] / 1000
    return epochs


def load_beh_data_one_subj(subj, eeg_data, data_dir=DATA_DIR, exp_num=EXP_NUM):
    """Load behavioral data for one subject"""
    # Load data from MAT file
    beh_mat_fname = os.path.join(data_dir, 'exp{}'.format(
        exp_num), 'Data', '{}_MixModel_wBias.mat'.format(subj))
    beh_data = read_mat(beh_mat_fname)['beh']['trial']

    # Remove trials with artifacts
    beh_data_cleaned = {k: val[~eeg_data['arf']['artIndCleaned'].astype(
        bool)] for k, val in beh_data.items()}
    return beh_data_cleaned
