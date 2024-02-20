"""Train and test IEM using the same methodology used by Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/)."""

# Import necessary modules
import os.path
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import mne
import params
from iem import IEM
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from astropy.stats import circorrcoef


def split_trials(
    subj,
    param,
    t_window,
    baseline_t_window=None,
    operation="-",
    channels=None,
    bottom_frac=0.5,
    top_frac=0.5,
    metadata=False,
    param_dir=params.SPARAM_DIR,
    task_timings=params.TASK_TIMINGS,
    subjects_by_task=params.SUBJECTS_BY_TASK,
):
    "Split trials based on given criterion."
    # Load parameterized data
    epochs, times, _, param_data = load_param_data(
        subj, param, param_dir, metadata=metadata
    )

    # Parse subject ID
    experiment, subj_num = subj.split("_")
    task_num = np.argmax(
        [
            exp == experiment and int(subj_num) in ids
            for exp, ids in subjects_by_task
        ]
    )

    # Chunk by time window if desired
    processed_data = param_data
    if t_window is not None:
        windowed_data = []
        for one_t_window in (baseline_t_window, t_window):
            # Skip if no time window
            if one_t_window is None:
                continue

            # Parse time window
            if one_t_window == "baseline":
                one_t_window = (times[0], 0.0)
            elif one_t_window == "stimulus":
                one_t_window = (0.0, task_timings[task_num][1])
            elif one_t_window == "delay":
                one_t_window = task_timings[task_num]
            elif one_t_window == "response":
                one_t_window = (task_timings[task_num][1], times[-1])

            # Average across time window
            t_window_idx = np.where(
                (times >= one_t_window[0]) & (times <= one_t_window[1])
            )[0]
            windowed_data.append(
                np.mean(param_data[:, :, t_window_idx], axis=-1)
            )

        # Perform operation between time windows
        processed_data = windowed_data[0]
        if len(windowed_data) == 2:
            if operation == "-":
                processed_data = windowed_data[1] - windowed_data[0]
            elif operation == "/":
                processed_data = windowed_data[1] / windowed_data[0]

    # Subset channels if desired
    if channels is not None:
        ch_mask = np.isin(epochs.ch_names, channels)
        processed_data = processed_data[:, ch_mask]

    # Average across channels if 2D and across channels and time if 3D
    if processed_data.ndim > 1:
        axis = tuple(-np.arange(processed_data.ndim)[1:])
        processed_data = np.mean(processed_data, axis=axis)

    # Remove None values
    processed_data = np.array(list(filter(None, processed_data)))

    # Sort trials by parameterized data
    sorted_idx = np.argsort(processed_data)
    np.save(f"error.npy", processed_data)

    # Split trials into top and bottom percentiles
    n_trials = len(sorted_idx)
    n_bottom_trials = int(n_trials * bottom_frac)
    n_top_trials = int(n_trials * top_frac)
    trials_low = sorted_idx[:n_bottom_trials]
    trials_high = sorted_idx[-n_top_trials:]
    return trials_high, trials_low


def load_param_data(
    subj,
    param,
    param_dir,
    metadata=False,
    epochs_dir=params.EPOCHS_DIR,
    sparam_dir=params.SPARAM_DIR,
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

    # Extract position data
    pos_data = epochs.metadata["pos_bin"].values
    pos_nan = np.isnan(pos_data)

    # Load any skipped trials
    skipped_fname = f"{sparam_dir}/skipped.csv"
    if os.path.exists(skipped_fname):
        skipped_df = pd.read_csv(skipped_fname)
        skipped_dct = skipped_df.to_dict(orient="list")
        if subj in skipped_dct.keys():
            pos_nan[skipped_dct[subj]] = True

    # Remove trials with NaNs
    pos_data = pos_data[~pos_nan]
    epochs.drop(pos_nan, verbose=False)

    # Get times from epochs, taking account of decimation if applied
    times = epochs.times
    if param != "total_power":
        times = epochs.times[::decim_factor]

    # If metadata, grab from epochs object
    if metadata:
        param_data = epochs.metadata[param].values
        return epochs, times, pos_data, param_data

    # Load parameterized data
    param_data = mne.read_epochs(
        os.path.join(param_dir, f"{subj}_{param}_epo.fif"), verbose=False
    ).get_data(copy=True)
    param_data = param_data[~pos_nan, :, :]
    return epochs, times, pos_data, param_data


def equalize_param_data_across_trial_blocks(
    epochs, times, pos_data, param_data, average=True, n_blocks=params.N_BLOCKS
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
    pos_data : dict
        Dictionary containing positional data.
    param_data : np.ndarray
        Array containing parameterized data for decoding.
    average : bool (default: True)
        Whether to average across trials for each location bin.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to split trials into.

    Returns
    -------
    param_arr : np.ndarray
        Array containing averaged parameterized data for decoding.
    """
    # Extract relative variables from data
    assert np.count_nonzero(np.isnan(pos_data)) == 0
    n_bins = np.sum(~np.isnan(np.unique(pos_data)))
    n_channels = epochs.get_data(copy=True).shape[-2]
    n_timepts = len(times)

    # Determine number of trials per location bin
    _, counts = np.unique(pos_data, return_counts=True)
    n_trials_per_bin_per_block = counts.min() // n_blocks
    n_trials_per_bin = n_trials_per_bin_per_block * n_blocks
    idx_split_by_vals = np.split(
        np.argsort(pos_data), np.where(np.diff(sorted(pos_data)))[0] + 1
    )

    # Calculate parameterized data for block of trials
    n_segments = n_bins
    if not average:
        n_segments *= n_trials_per_bin_per_block
    param_arr = np.zeros((n_segments, n_blocks, n_channels, n_timepts))

    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Split parameterized data across block of trials
        split_data = param_data[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts
        )

        # Average across trials for each location bin if desired
        if average:
            split_data = np.nanmean(split_data, axis=0)

        # Take only real component of data
        blocked_data = np.real(split_data)

        # Add block-averaged parameterized data to array
        start_idx = i * n_trials_per_bin_per_block
        end_idx = (i + 1) * n_trials_per_bin_per_block
        if average:
            start_idx, end_idx = i, i + 1
        param_arr[start_idx:end_idx, :, :, :] = blocked_data

    # Rearrange axes of data to work for IEM pipeline
    param_arr = np.swapaxes(param_arr, 0, 2)
    return param_arr


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
    multi_output.fit(x_train, y_train_multi)

    # Transform the prediction back to angle
    prediction = multi_output.predict(x_test)
    y_predicted = np.arctan2(prediction[:, 0], prediction[:, 1])

    # Compute the circular correlation coefficient
    circ_corr = circorrcoef(y_test, y_predicted)
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
    n_timepts,
    n_blocks=params.N_BLOCKS,
    method="iem",
    single_trials=False,
):
    # Iterate through blocks
    model_fit = np.zeros((n_blocks, n_timepts))
    model_fit_null = np.zeros((n_blocks, n_timepts))
    for test_block_num in range(n_blocks):
        # Split into training and testing data
        train_data = np.delete(param_arr, test_block_num, axis=1).reshape(
            param_arr.shape[0], -1, param_arr.shape[-1]
        )
        test_data = param_arr[:, test_block_num, :, :]

        # Create labels for training and testing
        base_labels = IEM().channel_centers
        if single_trials:
            # Infer number of trials per location bin per block
            n_trials_per_bin = test_data.shape[1] // len(base_labels)
            base_labels = np.repeat(base_labels, n_trials_per_bin)
        train_labels = np.tile(base_labels, n_blocks - 1)
        test_labels = base_labels

        # Train IEMs for block of data, with no shuffle
        model_fit[test_block_num, :] = train_and_test_one_block(
            train_data, train_labels, test_data, test_labels, method=method
        )

        # Train IEMs for block of data, with shuffle
        train_labels_shuffled = np.random.permutation(train_labels)
        model_fit_null[test_block_num, :] = train_and_test_one_block(
            train_data,
            train_labels_shuffled,
            test_data,
            test_labels,
            method=method,
        )
    return model_fit, model_fit_null


def train_and_test_one_subj(
    subj,
    param,
    param_dir,
    trial_split_criterion=None,
    n_block_iters=params.N_BLOCK_ITERS,
    output_dir=params.IEM_OUTPUT_DIR,
    method="iem",
    single_trials=False,
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
    epochs, times, pos_data, param_data = load_param_data(
        subj,
        param,
        param_dir,
    )

    # Split trials based on selection criteria
    pos_data_set = (pos_data,)
    param_data_set = (param_data,)
    tags = ("",)
    if trial_split_criterion is not None:
        tags = ("high", "low")
        trials_high, trials_low = split_trials(subj, **trial_split_criterion)
        split_param = trial_split_criterion["param"]

        # Skip processing if not enough trials in one category
        if len(trials_high) < 50 or len(trials_low) < 50:
            n_trials_str = f"(n = {len(trials_high) + len(trials_low)} trials)"
            print(
                f"Not enough {split_param} data for trial split with {subj} "
                + n_trials_str
            )
            return

        # Split data based on trial split
        pos_data_set = (
            pos_data[trials_high],
            pos_data[trials_low],
        )
        param_data_set = (
            param_data[trials_high],
            param_data[trials_low],
        )

        # Make directories specific to parameter and trial split (if there is
        # one)
        output_dir = f"{output_dir}/{split_param}"
        if "t_window" in trial_split_criterion:
            split_t_window = trial_split_criterion["t_window"]
            if split_t_window is not None:
                output_dir = f"{output_dir}_{split_t_window}"
        split_baseline = None
        if "baseline_t_window" in trial_split_criterion:
            split_baseline = trial_split_criterion["baseline_t_window"]
        if split_baseline is not None:
            operation = "diff"
            if "operation" in trial_split_criterion:
                operation = trial_split_criterion["operation"]
            output_dir = f"{output_dir}_{operation}_{split_baseline}"

    did_fitting = False
    for tag, pos_data, param_data in zip(tags, pos_data_set, param_data_set):
        # Make directories specific to parameter and trial split (if there is one)
        save_dir = os.path.join(output_dir, param, tag)
        os.makedirs(save_dir, exist_ok=True)

        # Determine file names
        model_fit_fname = f"{save_dir}/model_fit_{subj}.npy"
        model_fit_null_fname = f"{save_dir}/model_fit_null_{subj}.npy"

        # Skip processing if already done
        if os.path.exists(model_fit_fname) and os.path.exists(
            model_fit_null_fname
        ):
            continue

        # Iterate through sets of blocks
        did_fitting = True
        n_timepts = len(times)

        # Average parameterized data within trial blocks
        param_arrs = [
            equalize_param_data_across_trial_blocks(
                epochs, times, pos_data, param_data, average=not single_trials
            )
            for _ in range(n_block_iters)
        ]

        # Train and test IEMs for each block iteration using multiprocessing
        model_fit_lst, model_fit_null_lst = [], []
        for param_arr in param_arrs:
            one_block_set = train_and_test_one_block_set(
                param_arr,
                n_timepts,
                method=method,
                single_trials=single_trials,
            )
            model_fit_lst.append(one_block_set[0])
            model_fit_null_lst.append(one_block_set[1])
        model_fit = np.stack(model_fit_lst)
        model_fit_null = np.stack(model_fit_null_lst)

        # Average across blocks and block iterations
        mean_model_fit = np.mean(model_fit, axis=(0, 1))
        mean_model_fit_null = np.mean(model_fit_null, axis=(0, 1))

        # Save data to avoid unnecessary re-processing
        np.save(model_fit_fname, mean_model_fit)
        np.save(model_fit_null_fname, mean_model_fit_null)

    # Print processing time if desired
    if verbose and did_fitting:
        print(
            f"Fit IEMs on {param} data for {subj} in "
            f"{time.time() - start:.3f} s"
        )
    return


def _get_subject_list(
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
    trial_split_criterion=None,
    method="iem",
    output_dir=params.IEM_OUTPUT_DIR,
    single_trials=False,
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
    subjs = _get_subject_list(param, param_dir)

    # Process each subject's data
    for subj in subjs:
        # Train and test for one subject
        train_and_test_one_subj(
            subj,
            param,
            param_dir,
            trial_split_criterion=trial_split_criterion,
            method=method,
            output_dir=output_dir,
            single_trials=single_trials,
        )


def load_model_fits_one_subj(
    subj,
    param,
    param_dir,
    trial_split_criterion=None,
    output_dir=params.IEM_OUTPUT_DIR,
    verbose=True,
):
    """Load model fit data for one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to use for decoding.
    param_dir : str
        Directory containing parameterized data.
    trial_split_criterion : dict (default: None)
        Dictionary containing trial split criterion.
    output_dir : str (default: params.IEM_OUTPUT_DIR)
        Directory to load output from.
    verbose : bool (default: True)
        Whether to print processing time.

    Returns
    -------
    mean_model_fit : np.ndarray
        Array containing output from model fitting across time.
    mean_model_fit_null : np.ndarray
        Array containing null output from model fitting across time.
    """
    # Start timer
    start = time.time()

    # Load parameterized data
    _, times, _, _ = load_param_data(subj, param, param_dir)

    # Split trials based on selection criteria if desired
    tags = ("",)
    if trial_split_criterion is not None:
        tags = ("high", "low")
        split_param = trial_split_criterion["param"]
        output_dir = f"{output_dir}/{split_param}"
        if "t_window" in trial_split_criterion:
            split_t_window = trial_split_criterion["t_window"]
            if split_t_window is not None:
                output_dir = f"{output_dir}_{split_t_window}"
        split_baseline = None
        if "baseline_t_window" in trial_split_criterion:
            split_baseline = trial_split_criterion["baseline_t_window"]
        if split_baseline is not None:
            operation = "diff"
            if "operation" in trial_split_criterion:
                operation = trial_split_criterion["operation"]
            output_dir = f"{output_dir}_{operation}_{split_baseline}"

    mean_model_fit_dct, mean_model_fit_null_dct = {}, {}
    for tag in tags:
        # Determine file names
        load_dir = os.path.join(output_dir, param, tag)
        model_fit_fname = f"{load_dir}/model_fit_{subj}.npy"
        model_fit_null_fname = f"{load_dir}/model_fit_null_{subj}.npy"

        # Return if file does not exist
        if not os.path.exists(model_fit_fname):
            return None, None, None

        # Load slope array
        mean_model_fit = np.load(model_fit_fname)

        # Load null slope array
        mean_model_fit_null = np.load(model_fit_null_fname)

        # Put into dictionaries
        mean_model_fit_dct[tag] = mean_model_fit
        mean_model_fit_null_dct[tag] = mean_model_fit_null

    # Print processing time if desired
    if verbose:
        print(
            f"Loading {param} data for {subj} took "
            f"{time.time() - start:.3f} s"
        )

    if len(tags) > 1:
        return mean_model_fit_dct, mean_model_fit_null_dct, times
    return mean_model_fit, mean_model_fit_null, times


def load_all_model_fits(
    param,
    param_dir,
    trial_split_criterion=None,
    subjects_by_task=params.SUBJECTS_BY_TASK,
    output_dir=params.IEM_OUTPUT_DIR,
):
    """Load model fits for all subjects.

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
    subjs = _get_subject_list(param, param_dir)

    # Collate outputs from model fitting across subjects
    mean_model_fits = [[] for _ in range(len(subjects_by_task))]
    mean_model_fits_null = [[] for _ in range(len(subjects_by_task))]
    t_arrays = [[] for _ in range(len(subjects_by_task))]
    for subj in subjs:
        # Add data to big arrays
        mean_model_fit, mean_model_fit_null, t_arr = load_model_fits_one_subj(
            subj,
            param,
            param_dir,
            trial_split_criterion=trial_split_criterion,
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
        mean_model_fits_null[task_num].append(mean_model_fit_null)
        t_arrays[task_num] = t_arr

    # Collate outputs from model fitting across subjects
    mean_model_fits = [np.array(model_fit) for model_fit in mean_model_fits]
    mean_model_fits_null = [
        np.array(model_fit_null) for model_fit_null in mean_model_fits_null
    ]

    return (
        mean_model_fits,
        mean_model_fits_null,
        t_arrays,
    )


def fit_model_desired_params(
    sp_params="all",
    sparam_dir=params.SPARAM_DIR,
    total_power_dir=params.TOTAL_POWER_DIR,
    trial_split_criterion=None,
    method="iem",
    output_dir=params.IEM_OUTPUT_DIR,
    single_trials=False,
    verbose=True,
):
    """Fit desired model for total power and all parameters from spectral
    parameterization."""
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
            trial_split_criterion=trial_split_criterion,
            method=method,
            output_dir=output_dir,
            single_trials=single_trials,
        )
        if verbose:
            print(f"Fit IEMs for {sp_param} in {time.time() - start:.2f} s")

    # Load fit IEMs for all parameters from spectral parameterization
    model_fits_all_params = {}
    model_fits_null_all_params = {}
    t_all_params = {}
    for sp_param in sp_params:
        # Determine directory containing parameterized data
        param_dir = sparam_dir
        if sp_param == "total_power":
            param_dir = total_power_dir

        # Load fit IEMs for one parameter
        (
            model_fits_one_param,
            model_fits_null_one_param,
            t,
        ) = load_all_model_fits(
            sp_param,
            param_dir,
            trial_split_criterion=trial_split_criterion,
            output_dir=output_dir,
        )

        # Put into dictionaries
        model_fits_all_params[sp_param] = model_fits_one_param
        model_fits_null_all_params[sp_param] = model_fits_null_one_param
        t_all_params[sp_param] = t
    return model_fits_all_params, model_fits_null_all_params, t_all_params
