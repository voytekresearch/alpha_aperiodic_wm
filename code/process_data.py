"""Process MNE Epochs data for each subject."""

# Import necessary modules
import os
import os.path
import time
import warnings
import mne
import ray
import numpy as np
from fooof import FOOOFGroup
from fooof.analysis import get_band_peak_fg
import pandas as pd
import params


def compute_total_power(epochs, save_fname, alpha_band=params.ALPHA_BAND):
    """Following Foster et al. (2015), filter data in desired alpha band, apply
    Hilbert transform, and compute total power from analytic signal.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to compute total power for.
    save_fname : str
        Filename to save total power to.
    alpha_band : tuple (default: params.ALPHA_BAND)
        Alpha band to filter data in.

    Returns:
    --------
    total_power : mne.Epochs
        Epochs object containing total power data.
    """
    # Load total power data if already computed
    if os.path.exists(save_fname):
        return

    # Band-pass filter in alpha band
    alpha_band = epochs.copy().filter(*alpha_band)

    # Apply Hilbert transform to get analytic signal
    analytic_sig = alpha_band.copy().apply_hilbert()

    # Get total power from analytic signal
    total_power = analytic_sig.copy().apply_function(np.abs).apply_function(
        np.square)

    # Save data to avoid re-processing
    total_power.save(save_fname)


@ray.remote
def run_decomp_and_sparam_one_trial(
        epochs, trial_num, fmin=params.FMIN, fmax=params.FMAX,
        n_freqs=params.N_FREQS,time_window_len=params.TIME_WINDOW_LEN,
        decim_factor=params.DECIM_FACTOR, n_peaks=params.N_PEAKS,
        peak_width_lims=params.PEAK_WIDTH_LIMS, freq_band=params.ALPHA_BAND,
        verbose=True):
    """
    For one trial of data, run spectral decomposition and spectral
    parameterization.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to run spectral decomposition and spectral
        parameterization.
    trial_num : int
        Trial number to run spectral decomposition and spectral
        parameterization for.
    fmin : float (default: params.FMIN)
        Minimum frequency to use for spectral decomposition.
    fmax : float (default: params.FMAX)
        Maximum frequency to use for spectral decomposition.
    n_freqs : int (default: params.N_FREQS)
        Number of frequencies to use for spectral decomposition.
    time_window_len : float (default: params.TIME_WINDOW_LEN)
        Length of time window to use for spectral decomposition.
    decim_factor : int (default: params.DECIM_FACTOR)
        Decimation factor to use for spectral decomposition.
    n_peaks : int (default: params.N_PEAKS)
        Maximum number of peaks to use for spectral parameterization.
    peak_width_lims : tuple (default: params.PEAK_WIDTH_LIMS)
        Peak width limits to use for spectral parameterization.
    freq_band : tuple (default: params.ALPHA_BAND)
        Frequency band to extract peaks from spectral parameterization.
    verbose : bool (default: True)
        Whether to print runtime information.
    """
    # Start timer for spectral decomposition
    start = time.time()
    if verbose:
        print(f'Started spectral decomposition of trial #{trial_num}')

    # Make frequencies log-spaced
    freqs = np.linspace(fmin, fmax, n_freqs)

    # Make time window length consistent across frequencies
    n_cycles = freqs * time_window_len

    # Get current trial of data
    trial = epochs.copy().drop(
        [i for i in range(epochs.get_data().shape[0]) if i != trial_num],
        verbose=False)

    # Use multitapers to estimate spectrogram
    trial_tfr = mne.time_frequency.tfr_multitaper(
        trial, freqs, n_cycles, return_itc=False, picks='eeg', average=False,
        decim=decim_factor, verbose=False)

    # Reshape spectrogram to be (n_channels, n_timepts, n_freqs)
    tfr_arr = np.squeeze(np.swapaxes(trial_tfr.data, 2, 3))

    # Print runtime for spectral decomposition
    if verbose:
        print(f'Finished spectral decomposition for trial #{trial_num} in '
              f'{time.time() - start:.1f} seconds')

    # Start timer for spectral parameterization
    start = time.time()
    if verbose:
        print(f'Started spectral parameterization of trial #{trial_num}')

    # Initialize FOOOFGroup
    fooof_grp = FOOOFGroup(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_lims, verbose=False)

    # Fit spectral parameterization model
    fooof_grp.fit(
        freqs, tfr_arr.reshape(-1, len(freqs)), freq_range=(fmin, fmax))

    # Extract aperiodic and model fit parameters from model
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aperiodic_params = fooof_grp.get_params('aperiodic_params')
        r_squared = fooof_grp.get_params('r_squared')
        error = fooof_grp.get_params('error')

        # Select only peak parameters with peak frequency in desired frequency
        # band
        peak_params = get_band_peak_fg(fooof_grp, freq_band)

    # Put all parameters together
    model_params = np.hstack((
        aperiodic_params, peak_params, r_squared[:, np.newaxis],
        error[:, np.newaxis]))

    # Create DataFrame for trial
    n_channels, n_timepts, _ = tfr_arr.shape
    index_shape = (1, n_channels, n_timepts)
    index_names = ['trial', 'channel', 'timepoint']
    index = pd.MultiIndex.from_product([
        range(s) for s in index_shape], names=index_names)
    column_names = ['offset', 'exponent', 'CF', 'PW', 'BW', 'R^2', 'error']
    sparam_df_one_trial = pd.DataFrame(
        model_params, columns=column_names, index=index).reset_index()
    sparam_df_one_trial.loc[:, 'trial'] = trial_num

    # Print runtime for spectral parameterization
    if verbose:
        print(f'Finished spectral parameterization for trial #{trial_num} in '
              f'{time.time() - start:.1f} seconds')
    return trial_num, sparam_df_one_trial


def run_decomp_and_sparam_all_trials(epochs, save_dir):
    """For each trial of data, run spectral decomposition and spectral
    parameterization using ray for parallelization.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object containing data to be processed.
    save_dir : str
        Directory to save results to.

    Returns:
    --------
    sparam_df : pd.DataFrame
        DataFrame containing spectral parameterization results for all trials.
    """
    # Determine which trials have already been computed
    trials_computed = [int(f.split('.')[0].split('l')[-1]) for f in os.listdir(
        save_dir)]

    # Determine number of trials
    n_trials = epochs.get_data().shape[0]

    # If not all trials have been computed, compute remaining trials
    if not len(trials_computed) == n_trials:
        # Print remaining trials to compute
        trials_to_process = sorted(list(set(range(n_trials)) - set(
            trials_computed)))
        print(f'Already processed: {len(trials_computed)}\n'
            f'Still to process: {len(trials_to_process)}\n')

        # Iterate through each trial of data
        ray.init(address=os.environ["ip_head"])
        epochs_id = ray.put(epochs)

        # Fit spectral parameterization model for one trial
        result_ids = [run_decomp_and_sparam_one_trial.remote(
            epochs_id, trial_num) for trial_num in trials_to_process]

        # Save trial data as processed
        while result_ids:
            done_id, result_ids = ray.wait(result_ids)

            # Save trial DataFrame
            trial_num, sparam_df_one_trial = ray.get(done_id[0])
            save_fname = f'{save_dir}/sparam_trial{trial_num}.csv'
            sparam_df_one_trial.to_csv(save_fname, index=False)

        # Shut down ray
        ray.shutdown()

    # Concatenate all trial DataFrames
    sparam_df_all_trials = pd.DataFrame([])
    for fname in os.listdir(save_dir):
        sparam_df_one_trial = pd.read_csv(f'{save_dir}/{fname}')
        sparam_df_all_trials = pd.concat(
            [sparam_df_all_trials, sparam_df_one_trial], ignore_index=True)
    return sparam_df_all_trials


def convert_sparam_df_to_mne(sparam_df, info, save_fname):
    """Convert spectral parameterization DataFrame to series of MNE epochs, one
    for each relevant spectral parameterization model parameter.

    Parameters:
    -----------
    sparam_df : pd.DataFrame
        Dataframe containing spectral parameterization results for all trials.
    info : mne.Info
        Info object containing metadata for MNE epochs.
    save_fname : str
        Filename to save spectral parameterization results to.  This is used to
        determine the filename for the MNE epochs.
    """
    # Reorganize spectral parameterization DataFrame
    sparam_df = sparam_df.set_index(['trial', 'channel', 'timepoint'])
    if len(sparam_df) != np.prod(sparam_df.index.levshape):
        sparam_df = sparam_df.reindex(pd.MultiIndex.from_product(
            [np.arange(n) for n in sparam_df.index.levshape],
            names=['trial', 'channel', 'timepoint']))

    # Fill remaining NaNs with zero
    sparam_df = sparam_df.fillna(0)

    # Iterate through each model parameter
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


def process_one_subject(
        subject_fname, processed_dir=params.PROCESSED_DIR,
        total_power_dir=params.TOTAL_POWER_DIR, sparam_dir=params.SPARAM_DIR):
    """Load EEG and behavioral data and then perform preprocessing for one
    subject.

    Preprocessing #1: Following Foster et al. (2015), filter data in desired
    alpha band, apply Hilbert transform, and compute total power from analytic
    signal.

    Preprocessing #2: Use multitaper to estimate PSD with sliding window,
    spectral parameterization to fit periodic and aperiodic components, and then
    isolate aperiodic exponent and alpha oscillatory power.

    Parameters:
    -----------
    subject_fname : str
        Filename of subject's EEG data.
    processed_dir : str (default: params.PROCESSED_DIR)
        Directory to save processed data to.
    total_power_dir : str (default: params.TOTAL_POWER_DIR)
        Directory to save total power data to.
    sparam_dir : str (default: params.SPARAM_DIR)
        Directory to save spectral parameterization data to.
    """
    # Extract experiment and subject from filename
    experiment, subject = subject_fname.split('_')[:2]
    subject = int(subject)

    # Make directory to save data to if necessary
    os.makedirs(processed_dir, exist_ok=True)

    # Load subject's EEG data
    epochs_fname = f'{processed_dir}/{experiment}_{subject}_eeg_epo.fif'
    epochs = mne.read_epochs(epochs_fname)

    # Calculate total power
    os.makedirs(total_power_dir, exist_ok=True)
    total_power_fname = (
        f'{total_power_dir}/{experiment}_{subject}_total_power_epo.fif')
    compute_total_power(epochs, total_power_fname)

    # Compute and parameterize spectrogram
    subj_sparam_dir = f'{sparam_dir}/{experiment}_{subject}'
    os.makedirs(subj_sparam_dir, exist_ok=True)
    print(f'\nStarting spectral decomposition and parameterization for Subject '
          f'{experiment}_{subject}')
    sparam_df = run_decomp_and_sparam_all_trials(epochs, subj_sparam_dir)

    # Extract spectral parameters from model and convert to mne
    sparam_epo_fname = f'{sparam_dir}/{experiment}_{subject}_epo.fif'
    convert_sparam_df_to_mne(sparam_df, epochs.info, sparam_epo_fname)


def process_all_subjects(
        niceness=params.NICENESS, processed_dir=params.PROCESSED_DIR):
    """Load EEG and behavioral data and then perform preprocessing for all
    subjects.

    Parameters:
    -----------
    niceness : int (default: params.NICENESS)
        Niceness value to set for process.
    processed_dir : str (default: params.PROCESSED_DIR)
        Directory to save processed data to.
    """
    # Set niceness
    os.nice(niceness)

    # Process each subject's data
    subject_files = os.listdir(processed_dir)
    for subject_fname in subject_files:
        process_one_subject(subject_fname)


if __name__ == '__main__':
    process_all_subjects()
