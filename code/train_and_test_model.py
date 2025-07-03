"""Train and test IEM using the same methodology used by Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/)."""

# Import necessary modules
import os.path
import time
import numpy as np
import mne
import params
from iem import IEM
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from astropy.stats import circcorrcoef


def load_param_data(
    subj,
    param,
    param_dir,
    metadata=False,
    epochs_dir=params.EPOCHS_DIR,
    decim_factor=params.DECIM_FACTOR,
):
    """Load processed EEG and positional data for one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to decode.
    epochs_dir : str (default: params.EPOCHS_DIR)
        Path to directory containing processed data.
    decim_factor : int (default: params.DECIM_FACTOR)
        Decimation factor.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : np.ndarray
        Time course.
    pos_data : dict
        Dictionary containing positional data.
    param_data : np.ndarray
        Array containing parameterized data for decoding.
    """
    # Load epoched EEG data
    epochs = mne.read_epochs(
        os.path.join(epochs_dir, f"{subj}_eeg_epo.fif"),
        preload=True,
        verbose=False,
    )

    # Select only EEG channels
    picks = mne.pick_types(
        epochs.info,
        eeg=True,
        meg=False,
        eog=False,
        misc=False,
        exclude=["HEOG", "VEOG", "StimTrak"],
    )
    epochs = epochs.pick(picks)

    # Get times from epochs, taking account of decimation if applied
    times = epochs.times
    if param != "total_power":
        times = epochs.times[::decim_factor]

    # If metadata, grab from epochs object
    if metadata:
        param_data = epochs.metadata[param].values
        return epochs, times, param_data

    # Load parameterized data
    param_data = (
        mne.read_epochs(
            os.path.join(param_dir, f"{subj}_{param}_epo.fif"), verbose=False
        )
        .pick(picks)
        .get_data(copy=True)
    )
    return epochs, times, param_data


def equalize_param_data_across_trial_blocks(
    epochs,
    times,
    param_data,
    average=True,
    n_blocks=params.N_BLOCKS,
    distractors=False,
):
    """Equalize parameterized data across trials blocks such that there are an
    equal number of trials for each location bin. Then, average across trials
    for each location bin if desired.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : np.ndarray
        Time course.
    param_data : np.ndarray
        Array containing parameterized data for decoding.
    average : bool (default: True)
        Whether to average across trials for each location bin.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to split trials into.
    distractors : bool (default: False)
        Whether to include distractor data.

    Returns
    -------
    param_arr : np.ndarray
        Array containing averaged parameterized data for decoding.
    pos_arr : np.ndarray or None
        Array containing positional data or None if skipped due to NaN.
    """
    # Extract positional data from epochs
    pos_bins = epochs.metadata["pos_bin"].values
    pos_data = epochs.metadata["pos"].values
    if distractors:
        pos_bins = epochs.metadata["pos_bin_nt"].values
        pos_data = epochs.metadata["pos_nt"].values

    # Check for None values and skip processing if any are found
    if np.any(pos_bins == None) or np.any(pos_data == None):
        return None, None

    # Ensure positional data is valid
    assert 0 <= pos_bins.min() < 360, "Invalid values in pos_bins."
    assert 0 <= pos_data.min() < 360, "Invalid values in pos_data."

    # Proceed with the rest of the function as is...
    n_bins = np.sum(~np.isnan(np.unique(pos_bins)))
    n_channels = epochs.get_data(copy=True).shape[-2]
    n_timepts = len(times)

    # Determine number of trials per location bin
    _, counts = np.unique(pos_bins, return_counts=True)
    n_trials_per_bin_per_block = counts.min() // n_blocks
    n_trials_per_bin = n_trials_per_bin_per_block * n_blocks
    idx_split_by_vals = np.split(
        np.argsort(pos_bins), np.where(np.diff(sorted(pos_bins)))[0] + 1
    )

    # Initialize arrays for processed data
    n_segments = n_bins
    if not average:
        n_segments *= n_trials_per_bin_per_block
    param_arr = np.zeros((n_segments, n_blocks, n_channels, n_timepts))
    pos_arr = None
    if pos_data is not None:
        pos_arr = np.zeros((n_blocks, n_segments))

    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Split parameterized data across block of trials
        split_data = param_data[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts
        )
        if pos_arr is not None:
            split_pos = pos_data[idx].reshape(n_blocks, -1)

        # Average across trials for each location bin if desired
        if average:
            split_data = np.nanmean(split_data, axis=0)
            pos_arr = None

        # Take only real component of data
        blocked_data = np.real(split_data)

        # Add block-averaged parameterized data to array
        start_idx = i * n_trials_per_bin_per_block
        end_idx = (i + 1) * n_trials_per_bin_per_block
        if average:
            start_idx, end_idx = i, i + 1
        param_arr[start_idx:end_idx, :, :, :] = blocked_data
        if pos_arr is not None:
            pos_arr[:, start_idx:end_idx] = split_pos

    # Rearrange axes of data to work for IEM pipeline
    param_arr = np.swapaxes(param_arr, 0, 2)

    # Recast positional data as integers
    if pos_arr is not None:
        pos_arr = pos_arr.astype(int)
    return param_arr, pos_arr


def fit_iem(train_data, train_labels, test_data, test_labels):
    """Estimate channel tuning function (CTF) for one block of training and
    testing data at one time point.

    Parameters
    ----------
    train_data : np.ndarray
        Array containing training data.
    train_labels : np.ndarray
        Array containing training labels.
    test_data : np.ndarray
        Array containing testing data.
    test_labels : np.ndarray
        Array containing testing labels.

    Returns
    -------
    ctf_slope : np.ndarray
        Array containing CTF slope from fitted IEM.
    """
    # Initialize IEM instance
    iem = IEM()

    # Train IEM using training data
    iem.train_model(train_data, train_labels)

    # Test IEM using testing data
    iem.estimate_ctf(test_data, test_labels)

    # Compute mean CTF slope
    iem.compute_ctf_slope()
    return iem.ctf_slope


def circ_ridge_regression(x_train, y_train, x_test, y_test):
    """Perform circular ridge regression to predict spatial location from
    spectral parameters of the EEG data."""
    # Initialize the model
    model = Ridge()

    # Initialize the multi-output regressor
    multi_output = MultiOutputRegressor(model)

    # Transform the target variable to circular space
    y_train_multi = np.array([np.sin(y_train), np.cos(y_train)]).T

    # Fit the model
    multi_output.fit(x_train.T, y_train_multi)

    # Transform the prediction back to angle
    prediction = multi_output.predict(x_test.T)
    y_predicted = np.arctan2(prediction[:, 0], prediction[:, 1])

    # Compute the circular correlation coefficient
    circ_corr = circcorrcoef(y_test, y_predicted)
    return circ_corr


def train_and_test_one_timepoint(
    train_data, train_labels, test_data, test_labels, method="iem"
):
    """Train and test given method for one time point."""
    if method == "iem":
        return fit_iem(train_data, train_labels, test_data, test_labels)
    elif method == "crr":
        return circ_ridge_regression(
            train_data, train_labels, test_data, test_labels
        )


def train_and_test_one_block(
    train_data,
    train_labels,
    test_data,
    test_labels,
    method="iem",
    time_axis=-1,
):
    """Fit model for one block of training and testing data across all time
    points.

    Parameters
    ----------
    train_data : np.ndarray
        Array containing training data.
    train_labels : np.ndarray
        Array containing training labels.
    test_data : np.ndarray
        Array containing testing data.
    test_labels : np.ndarray
        Array containing testing labels.
    time_axis : int (default: -1)
        Axis of data array containing time points.  Default is -1, which
        corresponds to the last axis.

    Returns
    -------
    model_fit : np.ndarray
        Array containing output from model fitting.
    """
    # Organize data into lists of arrays for each time point
    data_one_tp = [
        list(np.rollaxis(data, time_axis)) for data in (train_data, test_data)
    ]

    # Train and test IEM for each time point
    model_fit = []
    for train_data_one_tp, test_data_one_tp in zip(*data_one_tp):
        model_fit_one_tp = train_and_test_one_timepoint(
            train_data_one_tp,
            train_labels,
            test_data_one_tp,
            test_labels,
            method=method,
        )
        model_fit.append(model_fit_one_tp)
    return np.array(model_fit)


def train_and_test_one_block_set(
    param_arr,
    pos_data,
    n_timepts,
    n_blocks=params.N_BLOCKS,
    method="iem",
    single_trials=False,
):
    # Iterate through blocks
    model_fit = np.zeros((n_blocks, n_timepts))
    for test_block_num in range(n_blocks):
        # Split into training and testing data
        train_data = np.delete(param_arr, test_block_num, axis=1).reshape(
            param_arr.shape[0], -1, param_arr.shape[-1]
        )
        test_data = param_arr[:, test_block_num, :, :]

        # Create labels for training and testing
        base_labels = IEM().channel_centers
        train_labels = np.tile(base_labels, n_blocks - 1)
        test_labels = base_labels
        if single_trials:
            # Infer number of trials per location bin per block
            train_labels = np.delete(
                pos_data, test_block_num, axis=0
            ).flatten()
            test_labels = pos_data[test_block_num].flatten()

        # Train IEMs for block of data, with no shuffle
        model_fit[test_block_num, :] = train_and_test_one_block(
            train_data, train_labels, test_data, test_labels, method=method
        )
    return model_fit


def train_and_test_one_subj(
    subj,
    param,
    param_dir,
    n_block_iters=params.N_BLOCK_ITERS,
    output_dir=params.IEM_OUTPUT_DIR,
    method="iem",
    single_trials=False,
    distractors=False,
    verbose=True,
):
    """Train and test one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to use for decoding.
    param_dir : str
        Directory containing parameterized data.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to use for cross-validation.
    n_block_iters : int (default: params.N_BLOCK_ITERS)
        Number of iterations to use for cross-validation.
    output_dir : str (default: params.IEM_OUTPUT_DIR)
        Directory to save output to.
    verbose : bool (default: True)
        Whether to print processing time.

    Returns
    -------
    model_fit : np.ndarray
        Array containing output from model fitting across time.
    times : np.ndarray
        Array containing time points.
    """
    # Start timer
    start = time.time()

    # Load parameterized data
    epochs, times, param_data = load_param_data(
        subj,
        param,
        param_dir,
    )

    # Make directories specific to parameter and trial split
    # (if there is one)
    save_dir = os.path.join(output_dir, param)
    os.makedirs(save_dir, exist_ok=True)

    # Determine file names
    model_fit_fname = f"{save_dir}/model_fit_{subj}.npy"

    # Skip processing if already done
    if os.path.exists(model_fit_fname):
        return

    # Iterate through sets of blocks
    n_timepts = len(times)

    # Average parameterized data within trial blocks
    param_arrs, pos_arrs = list(
        zip(
            *[
                equalize_param_data_across_trial_blocks(
                    epochs,
                    times,
                    param_data,
                    average=not single_trials,
                    distractors=distractors,
                )
                for _ in range(n_block_iters)
            ]
        )
    )

    # Skip processing if any block returned None
    if any(param_arr is None for param_arr in param_arrs):
        if verbose:
            print(f"Skipping {subj} due to None values in positional data.")
        return

    # Train and test IEMs for each block iteration using multiprocessing
    model_fit_lst = []
    for param_arr, pos_data in zip(param_arrs, pos_arrs):
        # Skip if no positional data and single trials decoding is desired
        if pos_data is None and single_trials:
            continue
        one_block_set = train_and_test_one_block_set(
            param_arr,
            pos_data,
            n_timepts,
            method=method,
            single_trials=single_trials,
        )
        model_fit_lst.append(one_block_set)

    # Return if no fitting was done
    if not model_fit_lst:
        return

    # Stack model fits across block iterations
    model_fit = np.stack(model_fit_lst)

    # Average across blocks and block iterations
    mean_model_fit = np.mean(model_fit, axis=(0, 1))

    # Save data to avoid unnecessary re-processing
    np.save(model_fit_fname, mean_model_fit)

    # Print processing time if desired
    if verbose:
        print(
            f"Fit IEMs on {param} data for {subj} in "
            f"{time.time() - start:.3f} s"
        )
    return


def get_subject_list(
    param, param_dir, subjects_by_task=params.SUBJECTS_BY_TASK
):
    # Get all subject IDs that were processed
    subjs_processed = sorted(
        [
            "_".join(f.split("_")[:2])
            for f in os.listdir(param_dir)
            if f"_{param}_" in f
        ]
    )

    # Get all subject IDs for IEMs, excluding those that were not
    # processed or excluded
    subjs = []
    for experiment, subj_ids in subjects_by_task:
        subjs.extend(
            ["_".join((experiment, str(subj_id))) for subj_id in subj_ids]
        )
    subjs = sorted(list(set(subjs) & set(subjs_processed)))
    return subjs


def train_and_test_all_subjs(
    param,
    param_dir,
    method="iem",
    output_dir=params.IEM_OUTPUT_DIR,
    single_trials=False,
    distractors=False,
):
    """Train and test for all subjects.

    Parameters
    ----------
    param : str
        Parameter to use for decoding.
    param_dir : str
        Directory containing parameterized data.
    fig_dir : str (default: params.FIG_DIR)
        Directory to save figures to.

    Returns
    -------
    mean_model_fits : np.ndarray
        Array containing outputs from model fitting across subjects.
    t_arr : np.ndarray
        Array containing time points.
    """
    # Get all subject IDs for IEMs, excluding those that were not
    # processed or excluded
    subjs = get_subject_list(param, param_dir)

    # Process each subject's data
    for subj in subjs:
        # Train and test for one subject
        train_and_test_one_subj(
            subj,
            param,
            param_dir,
            method=method,
            output_dir=output_dir,
            single_trials=single_trials,
            distractors=distractors,
        )


def load_model_fits_one_subj(
    subj,
    param,
    param_dir,
    output_dir=params.IEM_OUTPUT_DIR,
    verbose=True,
):
    """Load model fit data for one subject."""
    # Start timer
    start = time.time()

    # Load parameterized data
    _, times, _ = load_param_data(subj, param, param_dir)

    # Determine file names
    load_dir = os.path.join(output_dir, param)
    model_fit_fname = f"{load_dir}/model_fit_{subj}.npy"

    # Return if file does not exist
    if not os.path.exists(model_fit_fname):
        return None, None

    # Load slope array
    mean_model_fit = np.load(model_fit_fname)

    # Print processing time if desired
    if verbose:
        print(
            f"Loading {param} data for {subj} took "
            f"{time.time() - start:.3f} s"
        )
    return mean_model_fit, times


def load_all_model_fits(
    param,
    param_dir,
    subjects_by_task=params.SUBJECTS_BY_TASK,
    output_dir=params.IEM_OUTPUT_DIR,
):
    """Load model fits for all subjects."""
    # Get all subject IDs for IEMs, excluding those that were not
    # processed or excluded
    subjs = get_subject_list(param, param_dir)

    # Collate outputs from model fitting across subjects
    mean_model_fits = [[] for _ in range(len(subjects_by_task))]
    t_arrays = [[] for _ in range(len(subjects_by_task))]
    for subj in subjs:
        # Add data to big arrays
        mean_model_fit, t_arr = load_model_fits_one_subj(
            subj,
            param,
            param_dir,
            output_dir=output_dir,
        )

        # Skip if no data
        if mean_model_fit is None:
            continue

        # Parse subject ID
        experiment, subj_num = subj.split("_")
        task_num = np.argmax(
            [
                exp == experiment and int(subj_num) in ids
                for exp, ids in subjects_by_task
            ]
        )

        # Add data to big arrays
        mean_model_fits[task_num].append(mean_model_fit)
        t_arrays[task_num] = t_arr

    # Collate outputs from model fitting across subjects
    mean_model_fits = [np.array(model_fit) for model_fit in mean_model_fits]

    return mean_model_fits, t_arrays


def fit_model_desired_params(
    sp_params="all",
    sparam_dir=params.SPARAM_DIR,
    total_power_dir=params.TOTAL_POWER_DIR,
    method="iem",
    output_dir=params.IEM_OUTPUT_DIR,
    single_trials=False,
    distractors=False,
    verbose=True,
):
    """Fit desired model for total power and all parameters from spectral
    parameterization."""
    # If distractors, set output directory to reflect this
    if distractors:
        output_dir = f"{output_dir}_distractors"

    # Determine all parameters to fit IEM for
    if sp_params == "all":
        sp_params = {
            f.split("_")[-2]
            for f in os.listdir(sparam_dir)
            if f.endswith(".fif")
        }
        sp_params.add("total_power")

    # Fit IEMs for all parameters from spectral parameterization
    for sp_param in sp_params:
        # Start timer
        start = time.time()

        # Determine directory containing parameterized data
        param_dir = sparam_dir
        if sp_param == "total_power":
            param_dir = total_power_dir

        # Fit IEMs for one parameter
        train_and_test_all_subjs(
            sp_param,
            param_dir,
            method=method,
            output_dir=output_dir,
            single_trials=single_trials,
            distractors=distractors,
        )
        if verbose:
            print(f"Fit IEMs for {sp_param} in {time.time() - start:.2f} s")

    # Load fit IEMs for all parameters from spectral parameterization
    model_fits_all_params = {}
    t_all_params = {}
    for sp_param in sp_params:
        # Determine directory containing parameterized data
        param_dir = sparam_dir
        if sp_param == "total_power":
            param_dir = total_power_dir

        # Load fit IEMs for one parameter
        model_fits_one_param, t = load_all_model_fits(
            sp_param,
            param_dir,
            output_dir=output_dir,
        )

        # Put into dictionaries
        model_fits_all_params[sp_param] = model_fits_one_param
        t_all_params[sp_param] = t
    return model_fits_all_params, t_all_params
