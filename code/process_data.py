"""Load EEG and behavioral data from MAT files downloaded from OSF
(https://osf.io/bwzfj/)"""

# Import necessary modules
import os
import os.path
import mne
from mne.externals.pymatreader import read_mat
import numpy as np
import params


def load_eeg_data(
    subj, download_dir=params.DOWNLOAD_DIR, eeg_dir=params.EEG_DIR,
    experiment_num=params.EXPERIMENT_NUM):
    """Load EEG data for one subject."""
    # Load data from MAT file
    eeg_mat_fname = os.path.join(
        download_dir, f'exp{experiment_num}', eeg_dir, f'{subj}_EEG.mat')
    eeg_data = read_mat(eeg_mat_fname)['eeg']

    # Create epochs array from loaded MAT file
    info = mne.create_info(
        eeg_data['chanLabels'], eeg_data['sampRate'], ch_types='eeg')
    epochs = mne.EpochsArray(
        eeg_data['data'], info, tmin=eeg_data['preTime'] / 1000).drop(
            eeg_data['arf']['artIndCleaned'].astype(bool))
    assert epochs.times[-1] == eeg_data['postTime'] / 1000
    return eeg_data, epochs


def load_beh_data(
    subj, eeg_data, download_dir=params.DOWNLOAD_DIR,
    experiment_num=params.EXPERIMENT_NUM, cleaned=True):
    """Load behavioral data for one subject."""
    # Load data from MAT file
    beh_mat_fname = os.path.join(
        download_dir, f'exp{experiment_num}', 'Data',
        f'{subj}_MixModel_wBias.mat')
    beh_data = read_mat(beh_mat_fname)['beh']['trial']

    # Return all behavioral data if desired
    if not cleaned:
        return beh_data

    # Remove trials with artifacts
    beh_data_cleaned = {k: val[~eeg_data['arf']['artIndCleaned'].astype(
        bool)] for k, val in beh_data.items()}
    return beh_data_cleaned


def compute_tfr(subj, epochs, fmin=params.FMIN, fmax=params.FMAX,
        n_freqs=params.N_FREQS, time_window_len=params.TIME_WINDOW_LEN,
        processed_dir=params.PROCESSED_DIR):
    """Compute time-frequency representation (i.e. spectrogram) using
    multitapers across epochs and channels."""
    # Make file name for spectrogram to be saved with
    save_fname = os.path.join(processed_dir, f'{subj}-tfr.h5')

    # Load file if already computed
    if os.path.exists(save_fname):
        tfr_mt = mne.time_frequency.read_tfrs(save_fname)
        return tfr_mt

    # Make frequencies log-spaced
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)

    # Make time window length consistent across frequencies
    n_cycles = freqs * time_window_len

    # Use multitapers to estimate spectrogram
    tfr_mt = mne.time_frequency.tfr_multitaper(
        epochs.copy(), freqs, n_cycles, n_jobs=4, return_itc=False, picks='eeg',
        average=False)

    # Save spectrogram
    tfr_mt.save(save_fname)
    return tfr_mt


def compute_total_power(epochs, alpha_band=params.ALPHA_BAND):
    """Following Foster et al. (2015), filter data in desired alpha band, apply
    Hilbert transform, and compute total power from analytic signal.

    Then, use multitaper to estimate PSD with sliding window, spectral
    parameterization to fit periodic and aperiodic components, and then isolate
    aperiodic exponent and alpha oscillatory power."""
    # Band-pass filter in alpha band
    alpha_band = epochs.copy().filter(*alpha_band)

    # Apply Hilbert transform to get analytic signal
    analytic_sig = alpha_band.copy().apply_hilbert()

    # Get total power from analytic signal
    total_power = analytic_sig.copy().apply_function(np.abs).apply_function(
        np.square)
    return total_power


def process_one_subj(subj, processed_dir=params.PROCESSED_DIR):
    """Load EEG and behavioral data and then perform preprocessing for one
    subject.

    Preprocessing #1: Following Foster et al. (2015), filter data in desired
    alpha band, apply Hilbert transform, and compute total power from analytic
    signal.

    Preprocessing #2: Then, use multitaper to estimate PSD with sliding window,
    spectral parameterization to fit periodic and aperiodic components, and then
    isolate aperiodic exponent and alpha oscillatory power."""
    # Make directory to save data to if necessary
    os.makedirs(processed_dir, exist_ok=True)

    # Load subject's EEG data
    eeg_data, epochs = load_eeg_data(subj)

    # Load subject's behavioral data
    beh_data = load_beh_data(subj, eeg_data)

    # Calculate total power
    total_power = compute_total_power(epochs)

    # Compute spectrogram
    tfr_mt = compute_tfr(subj, epochs)

    # Save all data for subject
    epochs.save(os.path.join(processed_dir, f'{subj}_eeg_data_epo.fif'))
    np.savez(os.path.join(processed_dir, f'{subj}_beh_data.npz'), **beh_data)
    total_power.save(os.path.join(processed_dir, f'{subj}_total_power_epo.fif'))


def process_all_subjs(
    experiment_num=params.EXPERIMENT_NUM, eeg_dir=params.EEG_DIR,
    download_dir=params.DOWNLOAD_DIR):
    """Load EEG and behavioral data and then perform preprocessing for all
    subjects."""
    # Get all subject IDs
    subjs = sorted([f.split('_')[0] for f in os.listdir(os.path.join(
        download_dir, f'exp{experiment_num}', eeg_dir)) if 'REJECT' not in f])

    # Process each subject's data
    for subj in subjs:
        process_one_subj(subj)


if __name__ == '__main__':
    process_all_subjs()
