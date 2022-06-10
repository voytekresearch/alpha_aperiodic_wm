"""Load EEG and behavioral data from MAT files downloaded from OSF
(https://osf.io/bwzfj/)"""

# Import necessary modules
import os
import os.path
import time
import mne
from mne.externals.pymatreader import read_mat
import ray
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
        decim_factor=params.DECIM_FACTOR, n_cpus=params.N_CPUS):
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
        picks='eeg', average=False, decim=decim_factor)

    # Save spectrogram
    tfr_mt.save(save_fname)
    return tfr_mt


@ray.remote
def run_sparam_one_trial(
        tfr_arr, trial_num, freqs, n_peaks=params.N_PEAKS,
        peak_width_lims=params.PEAK_WIDTH_LIMS, fmin=params.FMIN,
        fmax=params.FMAX, freq_band=params.ALPHA_BAND, verbose=True):
    """Parameterize the neural power spectra for each time point in the
    spectrogram for one trial."""
    # Start timer
    start = time.time()
    print(f'Started spectral parameterization of Trial #{trial_num}')

    # Initialize FOOOFGroup
    fooof_grp = FOOOFGroup(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_lims)

    # Fit spectral parameterization model
    fooof_grp = combine_fooofs(fit_fooof_3d(
        fooof_grp, freqs, tfr_arr[trial_num, :, :, :], freq_range=(fmin, fmax)))

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
    _, n_channels, n_timepts, _ = tfr_arr.shape
    index_shape = (1, n_channels, n_timepts)
    index_names = ['trial', 'channel', 'timepoint']
    index = pd.MultiIndex.from_product([
        range(s) for s in index_shape], names=index_names)
    column_names = ['offset', 'exponent', 'CF', 'PW', 'BW', 'R^2', 'error']
    sparam_df_one_trial = pd.DataFrame(
        model_params, columns=column_names, index=index).reset_index()
    sparam_df_one_trial.loc[:, 'trial'] = trial_num

    # Print progress
    if verbose:
        print(f'Spectral parameterization for Trial #{trial_num} in '
              f'{time.time() - start} seconds\n')
    return sparam_df_one_trial


def run_sparam_all_trials(tfr, save_fname, n_cpus=params.N_CPUS,):
    """Parameterize the neural power spectra for each time point in the
    spectrogram. Spectrogram (tfr) should have shape of (n_trials, n_channels,
    n_freqs, n_timepoints)."""
    # Initialize big DataFrame
    sparam_df_all_trials = pd.DataFrame([])
    trials_computed = set([])

    # Load DataFrame if already generated
    if os.path.exists(save_fname):
        sparam_df_all_trials = pd.read_csv(save_fname, index_col=False)

        # Determine which trials have already been computed
        trials_computed = set(sparam_df_all_trials['trial'])

    # Make copy of spectrogram
    tfr = tfr.copy()

    # Reshape spectrogram
    n_trials = tfr.data.shape[0]
    tfr.data = np.swapaxes(tfr.data, 2, 3)

    # Iterate through each trial of data
    ray.init(num_cpus=n_cpus)
    tfr_arr_id = ray.put(tfr.data)

    # Fit spectral parameterization model for one trial
    result_ids = [run_sparam_one_trial.remote(
        tfr_arr_id, trial_num, tfr.freqs) for trial_num in range(n_trials) if
        trial_num not in trials_computed]

    # Concatenate trial model parameters and save as processed
    while result_ids:
        done_id, result_ids = ray.wait(result_ids)

        # Add fit model parameters to big DataFrame
        sparam_df_all_trials = pd.concat(
            (sparam_df_all_trials, ray.get(done_id[0])), ignore_index=True)

        # Save DataFrame
        sparam_df_all_trials.to_csv(save_fname, index=False)
    ray.shutdown()
    return sparam_df_all_trials


def convert_sparam_df_to_mne(tfr_mt, sparam_df, save_fname):
    """Convert spectral parameterization DataFrame to series of MNE epochs, one
    for each relevant spectral parameterization model parameter."""
    # Reorganize spectral parameterization DataFrame
    sparam_df = sparam_df.set_index(['trial', 'channel', 'timepoint'])
    if len(sparam_df) != np.prod(sparam_df.index.levshape):
        sparam_df = sparam_df.reindex(pd.MultiIndex.from_product(
            [np.arange(n) for n in sparam_df.index.levshape],
            names=['trial', 'channel', 'timepoint']))

    # Remove trials with any failed fits
    sparam_df_flat = sparam_df.reset_index()
    failed_model_fit_trials = sparam_df.loc[sparam_df.isna().all(
        axis=1)].reset_index()['trial'].unique()
    sparam_df = sparam_df_flat[~sparam_df_flat['trial'].isin(
        failed_model_fit_trials)].set_index(['trial', 'channel', 'timepoint'])
    print(sparam_df.loc[sparam_df.isna().all(axis=1)].shape)
    # Fill remaining NaNs with zero
    sparam_df = sparam_df.fillna(0)
    print(sparam_df.isna().sum())

    # Collapse frequencies of TFR
    info = tfr_mt.copy().average(dim='freqs').info

    for col in sparam_df.columns:
        # Determine filename for selected model parameter
        col_epochs_fname = save_fname.replace('_epo', f'_{col}_epo')

        # Avoid recomputing
        if os.path.exists(col_epochs_fname):
            continue

        # Make MNE Epochs for selected model parameter
        arr = sparam_df[col].values.reshape(sparam_df.index.levshape)
        epochs_arr = mne.EpochsArray(arr, info)

        # Save EpochArray
        epochs_arr.save(col_epochs_fname)
    return


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
    sparam_df = run_sparam_all_trials(tfr_mt, sparam_df_fname)

    # Extract spectral parameters from model and convert to mne
    sparam_epo_fname = os.path.join(processed_dir, f'{subj}_epo.fif')
    convert_sparam_df_to_mne(tfr_mt, sparam_df, sparam_epo_fname)


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
