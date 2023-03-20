"""Load EEG and behavioral data from MAT files acquired from Awh/Vogel lab."""

# Import necessary modules
import os
import os.path
import time
import mne
from frozendict import frozendict
from pymatreader import read_mat
import ray
import numpy as np
from fooof import FOOOFGroup, fit_fooof_3d
from fooof.objs import combine_fooofs
from fooof.analysis import get_band_peak_fg
import pandas as pd
import params


def _index_nested_object(nested, indices, subject, return_num_subjects=False):
    """Index nested Python object with tuple.

    Parameters:
    -----------
    nested : dict or list
        Nested Python object to index.
    indices : tuple
        Tuple of indices to index nested object with.
    subject : int
        Subject number to index with.
    return_num_subjects : bool (default: False)
        Whether to return number of subjects in nested object.

    Returns:
    --------
    nested : dict or list
        Nested Python object indexed with given indices.
    """
    # If given indices are not in fact indices, return them as the desired value
    if not isinstance(indices, tuple):
        return indices

    # Zoom in to next level
    if isinstance(nested, dict):
        if indices[0] not in nested.keys():
            return None
        nested = nested[indices[0]]
        indices = indices[1:]
    elif isinstance(nested, list):
        # Return number of subjects if desired
        if return_num_subjects:
            return len(nested)
        nested = nested[subject]

    # Return subject data if at end of nested indices
    if not indices:
        if isinstance(nested, list):
            # Return number of subjects if desired
            if return_num_subjects:
                return len(nested)
            if not isinstance(nested[0], str):
                nested = nested[subject]
        return nested

    # Continue with recursion if nested indices remain
    return _index_nested_object(nested, indices, subject, return_num_subjects)


def split_data_by_subject(
        experiment, experiment_vars=frozendict(params.EXPERIMENT_VARS),
        num_subjects=frozendict(params.NUM_SUBJECTS),
        download_dir=params.DOWNLOAD_DIR, processed_dir=params.PROCESSED_DIR):
    """Split data for experiments into individual subjects.

    Parameters:
    -----------
    experiment : str
        Experiment to split data for.
    experiment_vars : dict (default: params.EXPERIMENT_VARS)
        Dictionary of experiment variables.
    num_subjects : dict (default: params.NUM_SUBJECTS)
        Dictionary of number of subjects for each experiment.
    download_dir : str (default: params.DOWNLOAD_DIR)
        Directory containing downloaded data.
    processed_dir : str (default: params.PROCESSED_DIR)
        Directory to save processed data to.
    """
    # Make directory if necessary
    os.makedirs(processed_dir, exist_ok=True)

    # Restrict experiment variables to selected experiment
    num_subjects = num_subjects[experiment]
    experiment_vars = experiment_vars[experiment]

    # Check if experiment already processed to avoid loading large MAT file
    # if possible
    num_processed_files = len([f for f in os.listdir(
        processed_dir) if experiment in f])
    if num_subjects == num_processed_files / 2:
        return

    # Load data from experiment
    mat_fn = f'{download_dir}/spatialData_{experiment}.mat'
    exp_data = read_mat(mat_fn)

    # Process data for each subject
    for subject in range(num_subjects):
        # Extract data from big nested dictionary
        eeg_data = _index_nested_object(
            exp_data, experiment_vars['data'], subject)

        # Move on to next subject if no data found
        if eeg_data is None:
            print('No EEG data')
            continue

        # Extract relevant experimental variables from big nested dictionary
        ch_labels = _index_nested_object(
            exp_data, experiment_vars['ch_labels'], subject)
        bad_trials = _index_nested_object(
            exp_data, experiment_vars['bad_trials'], subject).astype(bool)
        sfreq = _index_nested_object(
            exp_data, experiment_vars['sfreq'], subject)
        pre_time = _index_nested_object(
            exp_data, experiment_vars['pre_time'], subject)
        post_time = _index_nested_object(
            exp_data, experiment_vars['post_time'], subject)
        pos_bin = _index_nested_object(
            exp_data, experiment_vars['pos_bin'], subject)
        bad_electrodes = _index_nested_object(
            exp_data, experiment_vars['bad_electrodes'], subject)
        art_pre_time = _index_nested_object(
            exp_data, experiment_vars['art_pre_time'], subject)
        art_post_time = _index_nested_object(
            exp_data, experiment_vars['art_post_time'], subject)

        # Create epochs array from loaded MAT file for each subject if needed
        epochs_fname = f'{processed_dir}/{experiment}_{subject}_eeg_epo.fif'
        if not os.path.exists(epochs_fname):
            # Create info, labeling bad electrodes
            info = mne.create_info(ch_labels, sfreq, ch_types='eeg')
            if bad_electrodes is None:
                bad_electrodes = []
            info['bads'] = [e for e in list(bad_electrodes) if e in ch_labels]

            # Turn data array into MNE EpochsArray with proper cropping applied
            epochs = mne.EpochsArray(
                eeg_data, info, tmin=-np.abs(pre_time)/1000)
            if epochs.times[-1] != post_time / 1000:
                epochs = mne.EpochsArray(
                    eeg_data, info, tmin=-np.abs(art_pre_time)/1000)
            else:
                # Crop data
                if art_pre_time:
                    epochs.crop(tmin=-np.abs(
                        art_pre_time)/1000, tmax=art_post_time/1000)
                    # Make sure times are consistent
                    assert epochs.times[-1] == art_post_time / 1000

            # Drop bad trials
            epochs = epochs.drop(bad_trials)

            # Save epochs
            epochs.save(epochs_fname)

        # Remove trials with artifacts
        beh_fname = f'{processed_dir}/{experiment}_{subject}_beh.npy'
        if not os.path.exists(beh_fname):
            beh_data = pos_bin.flat[~bad_trials]
            np.save(beh_fname, beh_data)


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
    multitapers across epochs and channels.

    Parameters:
    -----------
    epochs : mne.Epochs
        Epochs object to compute spectrogram for.
    save_fname : str
        Filename to save spectrogram to.
    fmin : float (default: params.FMIN)
        Minimum frequency to compute spectrogram for.
    fmax : float (default: params.FMAX)
        Maximum frequency to compute spectrogram for.
    n_freqs : int (default: params.N_FREQS)
        Number of frequencies to compute spectrogram for.
    time_window_len : float (default: params.TIME_WINDOW_LEN)
        Length of time window to use for each frequency.
    decim_factor : int (default: params.DECIM_FACTOR)
        Factor to downsample data by before computing spectrogram.
    n_cpus : int (default: params.N_CPUS)
        Number of CPUs to use for parallel processing.

    Returns:
    --------
    tfr_mt : mne.time_frequency.tfr_array.TFRArray
        Spectrogram data.
    """
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
    spectrogram for one trial.

    Parameters:
    -----------
    tfr_arr : np.ndarray
        Array of time-frequency representations (i.e. spectrograms) for each
        trial.
    trial_num : int
        Trial number to run spectral parameterization on.
    freqs : np.ndarray
        Array of frequencies corresponding to the spectrogram.
    n_peaks : int (default: params.N_PEAKS)
        Maximum number of peaks to fit in the model.
    peak_width_lims : tuple (default: params.PEAK_WIDTH_LIMS)
        Limits on peak width to fit in the model.
    fmin : float (default: params.FMIN)
        Minimum frequency to fit in the model.
    fmax : float (default: params.FMAX)
        Maximum frequency to fit in the model.
    freq_band : tuple (default: params.ALPHA_BAND)
        Frequency band to extract peak parameters from.
    verbose : bool (default: True)
        Whether to print progress.

    Returns:
    --------
    sparam_df_one_trial : pd.DataFrame
        Dataframe containing spectral parameterization results for one trial.
    """
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


def run_sparam_all_trials(tfr, save_fname, n_cpus=params.N_CPUS):
    """Parameterize the neural power spectra for each time point in the
    spectrogram.

    Parameters:
    -----------
    tfr : mne.time_frequency.tfr_array.TFRArray
        Time-frequency representation (i.e. spectrogram) for all trials. Shape
        should be (n_trials, n_channels, n_freqs, n_timepoints).
    save_fname : str
        Filename to save spectral parameterization results to.
    n_cpus : int (default: params.N_CPUS)
        Number of CPUs to use for parallelization.

    Returns:
    --------
    sparam_df_all_trials : pd.DataFrame
        Dataframe containing spectral parameterization results for all trials.
    """
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
    for each relevant spectral parameterization model parameter.

    Parameters:
    -----------
    tfr_mt : mne.time_frequency.tfr_array.TFRArray
        Time-frequency representation (i.e. spectrogram) for all trials. Shape
        should be (n_trials, n_channels, n_freqs, n_timepoints).
    sparam_df : pd.DataFrame
        Dataframe containing spectral parameterization results for all trials.
    save_fname : str
        Filename to save spectral parameterization results to.  This is used to
        determine the filename for the MNE epochs.

    Returns:
    --------
    epochs : mne.Epochs
        Epochs object containing spectral parameterization results for all
        trials.
    """
    # Reorganize spectral parameterization DataFrame
    sparam_df = sparam_df.set_index(['trial', 'channel', 'timepoint'])
    if len(sparam_df) != np.prod(sparam_df.index.levshape):
        sparam_df = sparam_df.reindex(pd.MultiIndex.from_product(
            [np.arange(n) for n in sparam_df.index.levshape],
            names=['trial', 'channel', 'timepoint']))

    # Fill remaining NaNs with zero
    sparam_df = sparam_df.fillna(0)

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


def process_one_subject(
        subject_fname, processed_dir=params.PROCESSED_DIR,
        total_power_dir=params.TOTAL_POWER_DIR, tfr_dir=params.TFR_DIR,
        sparam_dir=params.SPARAM_DIR):
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
    tfr_dir : str (default: params.TFR_DIR)
        Directory to save time-frequency representation (i.e. spectrogram) data
        to.
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
    total_power_fname = f'{total_power_dir}/{subject}_total_power_epo.fif'
    compute_total_power(epochs, total_power_fname)

    # Compute spectrogram
    tfr_fname = f'{tfr_dir}/{subject}-tfr.h5'
    tfr_mt = compute_tfr(epochs, tfr_fname)

    # Parameterize spectrogram
    sparam_df_fname = f'{sparam_dir}/{subject}_sparam.csv'
    sparam_df = run_sparam_all_trials(tfr_mt, sparam_df_fname)

    # Extract spectral parameters from model and convert to mne
    sparam_epo_fname = f'{sparam_dir}/{subject}_epo.fif'
    convert_sparam_df_to_mne(tfr_mt, sparam_df, sparam_epo_fname)


def process_all_subjects(
        niceness=params.NICENESS, download_dir=params.DOWNLOAD_DIR,
        processed_dir=params.PROCESSED_DIR):
    """Load EEG and behavioral data and then perform preprocessing for all
    subjects.

    Parameters:
    -----------
    niceness : int (default: params.NICENESS)
        Niceness value to set for process.
    download_dir : str (default: params.DOWNLOAD_DIR)
        Directory containing downloaded data.
    processed_dir : str (default: params.PROCESSED_DIR)
        Directory to save processed data to.
    """
    # Set niceness
    os.nice(niceness)

    # Split data by subjects if necessary
    experiments = set([f.split('.')[-2].split('_')[-1] for f in os.listdir(
        download_dir) if '.mat' in f])
    for experiment in experiments:
        split_data_by_subject(experiment)

    # Process each subject's data
    subject_files = os.listdir(processed_dir)
    for subject_fname in subject_files:
        process_one_subject(subject_fname)


if __name__ == '__main__':
    process_all_subjects()
