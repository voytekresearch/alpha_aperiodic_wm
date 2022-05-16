"""Load EEG and behavioral data from MAT files downloaded from OSF
(https://osf.io/bwzfj/)"""

# Import necessary modules
import os
import os.path
import mne
from mne.externals.pymatreader import read_mat
import numpy as np
from fooof import FOOOFGroup, fit_fooof_3d
from fooof.objs import combine_fooofs
from fooof.analysis import get_band_peak_fg
import pandas as pd
import params


def load_eeg_data(
    subj, epochs_fname, download_dir=params.DOWNLOAD_DIR,
    eeg_dir=params.EEG_DIR, experiment_num=params.EXPERIMENT_NUM):
    """Load EEG data for one subject."""
    # Load data from MAT file
    eeg_mat_fname = os.path.join(
        download_dir, f'exp{experiment_num}', eeg_dir, f'{subj}_EEG.mat')
    eeg_data = read_mat(eeg_mat_fname)['eeg']

    # Load epochs file if already created
    if os.path.exists(epochs_fname):
        return eeg_data, mne.read_epochs(epochs_fname)

    # Create epochs array from loaded MAT file
    info = mne.create_info(
        eeg_data['chanLabels'], eeg_data['sampRate'], ch_types='eeg')
    epochs = mne.EpochsArray(
        eeg_data['data'], info, tmin=eeg_data['preTime'] / 1000).drop(
            eeg_data['arf']['artIndCleaned'].astype(bool))
    assert epochs.times[-1] == eeg_data['postTime'] / 1000

    # Save epochs
    epochs.save(epochs_fname)
    return eeg_data, epochs


def load_beh_data(
    subj, eeg_data, save_fname, download_dir=params.DOWNLOAD_DIR,
    experiment_num=params.EXPERIMENT_NUM):
    """Load behavioral data for one subject."""
    # Load data if already computed
    if os.path.exists(save_fname):
        return np.load(save_fname)

    # Load data from MAT file
    beh_mat_fname = os.path.join(
        download_dir, f'exp{experiment_num}', 'Data',
        f'{subj}_MixModel_wBias.mat')
    beh_data = read_mat(beh_mat_fname)['beh']['trial']

    # Remove trials with artifacts
    beh_data_cleaned = {k: val[~eeg_data['arf']['artIndCleaned'].astype(
        bool)] for k, val in beh_data.items()}

    # Save data
    np.savez(save_fname, **beh_data_cleaned)
    return beh_data_cleaned


def compute_total_power(epochs, save_fname, alpha_band=params.ALPHA_BAND):
    """Following Foster et al. (2015), filter data in desired alpha band, apply
    Hilbert transform, and compute total power from analytic signal.

    Then, use multitaper to estimate PSD with sliding window, spectral
    parameterization to fit periodic and aperiodic components, and then isolate
    aperiodic exponent and alpha oscillatory power."""
    # Load total power data if already computed
    if os.path.exists(save_fname):
        return mne.read_epochs(save_fname)

    # Band-pass filter in alpha band
    alpha_band = epochs.copy().filter(*alpha_band)

    # Apply Hilbert transform to get analytic signal
    analytic_sig = alpha_band.copy().apply_hilbert()

    # Get total power from analytic signal
    total_power = analytic_sig.copy().apply_function(np.abs).apply_function(
        np.square)

    # Save data to avoid re-processing
    total_power.save(save_fname)
    return total_power


def compute_tfr(epochs, save_fname, fmin=params.FMIN, fmax=params.FMAX,
        n_freqs=params.N_FREQS, time_window_len=params.TIME_WINDOW_LEN,
        decim_fact=params.DECIM_FACT, n_cpus=params.N_CPUS):
    """Compute time-frequency representation (i.e. spectrogram) using
    multitapers across epochs and channels."""
    # Load file if already computed
    if os.path.exists(save_fname):
        tfr_mt = mne.time_frequency.read_tfrs(save_fname)
        return tfr_mt[0]

    # Make frequencies log-spaced
    freqs = np.linspace(fmin, fmax, n_freqs)

    # Make time window length consistent across frequencies
    n_cycles = freqs * time_window_len

    # Use multitapers to estimate spectrogram
    tfr_mt = mne.time_frequency.tfr_multitaper(
        epochs.copy(), freqs, n_cycles, n_jobs=n_cpus, return_itc=False,
        picks='eeg', average=False, decim=decim_fact)

    # Save spectrogram
    tfr_mt.save(save_fname)
    return tfr_mt


def run_sparam_one_trial(
        tfr_one_trial_arr, trial_num, freqs, n_peaks=params.N_PEAKS,
        peak_width_lims=params.PEAK_WIDTH_LIMS, fmin=params.FMIN,
        fmax=params.FMAX, n_cpus=params.N_CPUS, freq_band=params.ALPHA_BAND):
    """Parameterize the neural power spectra for each time point in the
    spectrogram for one trial."""
    # Initialize FOOOFGroup
    fooof_grp = FOOOFGroup(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_lims)

    # Fit spectral parameterization model
    fooof_grp = combine_fooofs(fit_fooof_3d(
        fooof_grp, freqs, tfr_one_trial_arr, freq_range=(fmin, fmax),
        n_jobs=n_cpus))

    # Extract aperiodic and model fit parameters from model
    aperiodic_params = fooof_grp.get_params('aperiodic_params')
    r_squared = fooof_grp.get_params('r_squared')
    error = fooof_grp.get_params('error')

    # Select only peak parameters with peak frequency in desired frequency band
    peak_params = get_band_peak_fg(fooof_grp, freq_band)

    # Put all parameters together
    model_params = np.hstack((
        aperiodic_params, peak_params, r_squared[:, np.newaxis],
        error[:, np.newaxis]))

    # Create DataFrame for trial
    n_channels, n_timepts, _ = tfr_one_trial_arr.shape
    index_shape = (1, n_channels, n_timepts)
    index_names = ['trial', 'channel', 'timepoint']
    index = pd.MultiIndex.from_product([
        range(s) for s in index_shape], names=index_names)
    column_names = ['offset', 'exponent', 'CF', 'PW', 'BW', 'R^2', 'error']
    sparam_df_one_trial = pd.DataFrame(
        model_params, columns=column_names, index=index).reset_index()
    sparam_df_one_trial['trial'] = trial_num
    return sparam_df_one_trial


def run_sparam_all_trials(tfr, save_fname):
    """Parameterize the neural power spectra for each time point in the
    spectrogram. Spectrogram (tfr) should have shape of (n_trials, n_channels,
    n_freqs, n_timepoints)."""
    # Load DataFrame if already generated
    if os.path.exists(save_fname):
        return pd.read_csv(save_fname, index_col=False)

    # Make copy of spectrogram
    tfr = tfr.copy()

    # Reshape spectrogram
    tfr.data = np.swapaxes(tfr.data, 2, 3)

    # Initialize big DataFrame
    sparam_df_all_trials = pd.DataFrame([])

    # Iterate through each trial of data
    for trial_num, trial_tfr in enumerate(tfr.data):
        # Fit spectral parameterization model for one trial
        sparam_df_one_trial = run_sparam_one_trial(
            trial_tfr, trial_num, tfr.freqs)

        # Add fit model parameters to big DataFrame
        sparam_df_all_trials = pd.concat(
            (sparam_df_all_trials, sparam_df_one_trial), ignore_index=True)

        # Save DataFrame
        sparam_df_all_trials.to_csv(save_fname, index=False)
    return sparam_df_all_trials


def process_one_subj(subj, processed_dir=params.PROCESSED_DIR):
    """Load EEG and behavioral data and then perform preprocessing for one
    subject.

    Preprocessing #1: Following Foster et al. (2015), filter data in desired
    alpha band, apply Hilbert transform, and compute total power from analytic
    signal.

    Preprocessing #2: Use multitaper to estimate PSD with sliding window,
    spectral parameterization to fit periodic and aperiodic components, and then
    isolate aperiodic exponent and alpha oscillatory power."""
    # Make directory to save data to if necessary
    os.makedirs(processed_dir, exist_ok=True)

    # Load subject's EEG data
    epochs_fname = os.path.join(processed_dir, f'{subj}_eeg_data_epo.fif')
    eeg_data, epochs = load_eeg_data(subj, epochs_fname)

    # Load subject's behavioral data
    beh_data_fname = os.path.join(processed_dir, f'{subj}_beh_data.npz')
    load_beh_data(subj, eeg_data, beh_data_fname)

    # Calculate total power
    total_power_fname = os.path.join(
        processed_dir, f'{subj}_total_power_epo.fif')
    compute_total_power(epochs, total_power_fname)

    # Compute spectrogram
    tfr_fname = os.path.join(processed_dir, f'{subj}-tfr.h5')
    tfr_mt = compute_tfr(epochs, tfr_fname)

    # Parameterize spectrogram
    sparam_df_fname = os.path.join(processed_dir, f'{subj}_sparam.csv')
    run_sparam_all_trials(tfr_mt, sparam_df_fname)


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
