"""Replicate decoding of spatial location from """

# Import neccesary modules
import os.path
import numpy as np
import mne
import params


def load_processed_data(subj, processed_dir=params.PROCESSED_DIR):
    """Load processed EEG and behavioral data for one subject."""
    # Load epoched EEG data
    epochs = mne.io.read_raw_fif(os.path.join(
        processed_dir, f'{subj}_eeg_data_epo.fif'))

    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f'{subj}_beh_data.npz'))

    # Load alpha total power data
    total_power = mne.io.read_raw_fif(os.path.join(
        processed_dir, f'{subj}_total_power_epo.fif'))
    return epochs, beh_data, total_power
