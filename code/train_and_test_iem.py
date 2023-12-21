"""Train and test IEM using the same methodology used by Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/)."""

# Import necessary modules
import os.path
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import mne
import params
from iem import IEM


def split_trials(
    subj,
    param_dir,
    param,
    t_window,
    baseline_t_window=None,
    operation="-",
    channels=None,
    bottom_pct=0.5,
    top_pct=0.5,
    task_timings=params.TASK_TIMINGS,
    subjects_by_task=params.SUBJECTS_BY_TASK,
):
    "Split trials based on given criterion."
    # Load parameterized data
    epochs, times, _, param_data = load_param_data(subj, param, param_dir)

    # Parse subject ID
    experiment, subj_num = subj.split("_")
    task_num = np.argmax(
        [
            exp == experiment and int(subj_num) in ids
            for exp, ids in subjects_by_task
        ]
    )

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
        windowed_data.append(np.mean(param_data[:, :, t_window_idx], axis=-1))

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

    # Average across channels
    processed_data = np.mean(processed_data, axis=-1)

    # Sort trials by parameterized data
    sorted_idx = np.argsort(processed_data)

    # Split trials into top and bottom percentiles
    n_trials = len(sorted_idx)
    n_bottom_trials = int(n_trials * bottom_pct)
    n_top_trials = int(n_trials * top_pct)
    trials_low = sorted_idx[:n_bottom_trials]
    trials_high = sorted_idx[-n_top_trials:]
    return trials_high, trials_low


def load_param_data(
    subj,
    param,
    param_dir,
    processed_dir=params.PROCESSED_DIR,
    sparam_dir=params.SPARAM_DIR,
    decim_factor=params.DECIM_FACTOR,
):
    """Load processed EEG and behavioral data for one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to decode.
    processed_dir : str (default: params.PROCESSED_DIR)
        Path to directory containing processed data.
    decim_factor : int (default: params.DECIM_FACTOR)
        Decimation factor.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : np.ndarray
        Time course.
    beh_data : dict
        Dictionary containing behavioral data.
    param_data : np.ndarray
        Array containing parameterized data for decoding.
    """
    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f"{subj}_beh.npy"))
    beh_nan = np.isnan(beh_data)

    # Load any skipped trials
    skipped_fname = f"{sparam_dir}/skipped.csv"
    if os.path.exists(skipped_fname):
        skipped_df = pd.read_csv(skipped_fname)
        skipped_dct = skipped_df.to_dict(orient="list")
        if subj in skipped_dct.keys():
            beh_nan[skipped_dct[subj]] = True

    # Remove trials with NaNs
    beh_data = beh_data[~beh_nan]

    # Load epoched EEG data
    epochs = mne.read_epochs(
        os.path.join(processed_dir, f"{subj}_eeg_epo.fif"),
        preload=True,
        verbose=False,
    )
    epochs.drop(beh_nan, verbose=False)

    # Get times from epochs, taking account of decimation if applied
    times = epochs.times
    if param != "total_power":
        times = epochs.times[::decim_factor]

    # Load parameterized data
    param_data = mne.read_epochs(
        os.path.join(param_dir, f"{subj}_{param}_epo.fif"), verbose=False
    ).get_data(copy=True)
    param_data = param_data[~beh_nan, :, :]
    return epochs, times, beh_data, param_data


def average_param_data_within_trial_blocks(
    epochs, times, beh_data, param_data, n_blocks=params.N_BLOCKS
):
    """Averaging parameterized data across trials within a block for each
    location bin.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : np.ndarray
        Time course.
    beh_data : dict
        Dictionary containing behavioral data.
    param_data : np.ndarray
        Array containing parameterized data for decoding.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to split trials into.

    Returns
    -------
    param_arr : np.ndarray
        Array containing averaged parameterized data for decoding.
    """
    # Extract relative variables from data
    assert np.count_nonzero(np.isnan(beh_data)) == 0
    n_bins = np.sum(~np.isnan(np.unique(beh_data)))
    n_channels = epochs.get_data(copy=True).shape[-2]
    n_timepts = len(times)

    # Determine number of trials per location bin
    _, counts = np.unique(beh_data, return_counts=True)
    n_trials_per_bin = counts.min() // n_blocks * n_blocks
    idx_split_by_vals = np.split(
        np.argsort(beh_data), np.where(np.diff(sorted(beh_data)))[0] + 1
    )

    # Calculate parameterized data for block of trials
    param_arr = np.zeros((n_blocks, n_bins, n_channels, n_timepts))

    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Average parameterized data across block of trials
        block_avg = np.real(
            np.nanmean(
                param_data[idx, :, :].reshape(
                    -1, n_blocks, n_channels, n_timepts
                ),
                axis=0,
            )
        )

        # Add block-averaged parameterized data to array
        param_arr[:, i, :, :] = block_avg

    # Rearrange axes of data to work for IEM pipeline
    param_arr = np.moveaxis(param_arr, 2, 0)
    return param_arr


def iem_one_timepoint(train_data, train_labels, test_data, test_labels):
    """Estimate channel response function (CRF) for one block of training and
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


def iem_one_block(
    train_data, train_labels, test_data, test_labels, time_axis=-1
):
    """Estimate channel response function (CRF) for one block of training and
    testing data across all time points.

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
    ctf_slope : np.ndarray
        Array containing CTF slope from fitted IEM.
    """
    # Organize data into lists of arrays for each time point
    data_one_tp = [
        list(np.rollaxis(data, time_axis)) for data in (train_data, test_data)
    ]

    # Make arguments for multiprocessing such that each function call will
    # contain training and testing data and labels for one time point
    args = tuple(
        [
            (train_data_one_tp, train_labels, test_data_one_tp, test_labels)
            for train_data_one_tp, test_data_one_tp in zip(*data_one_tp)
        ]
    )

    # Parallelize training and testing of IEM across time points
    with mp.Pool() as pool:
        ctf_slope = pool.starmap(iem_one_timepoint, args)
    return np.array(ctf_slope)


def plot_ctf_slope(
    ctf_slopes,
    t_arr,
    task_num,
    task_timings,
    ctf_slopes_shuffled=None,
    palette=None,
    save_fname=None,
    plot_timings=True,
    plot_errorbars=False,
    ax=None,
):
    """Plot channel tuning function (CTF) across time for multiple
    parameters.

    Parameters
    ----------
    ctf_slopes : dict
        Dictionary containing CTF slopes for each parameter. Keys are parameter
        names and values are arrays containing CTF slopes.
    t_arr : np.ndarray
        Array containing time points.
    palette : dict (default: None)
        Dictionary containing colors for each parameter.  Keys are parameter
        names and values are colors.  If None, default colors will be used.
    save_fname : str (default: None)
        File name to save figure to.  If None, figure will not be saved.
    """
    # Make empty list for CTF slope DataFrames
    ctf_slopes_dfs = []

    # Make DataFrame of CTF slopes by time for each parameter
    for param, ctf_slopes_one_param in ctf_slopes.items():
        n = ctf_slopes_one_param.shape[0]
        one_param_df = pd.DataFrame(ctf_slopes_one_param, columns=t_arr)
        one_param_df["Parameter"] = param
        one_param_df = one_param_df.melt(
            id_vars=["Parameter"], var_name="Time (s)", value_name="CTF slope"
        )
        one_param_df["Shuffled location labels?"] = "No"

        # Add DataFrame of CTF slopes for shuffled location labels if desired
        if ctf_slopes_shuffled is not None:
            one_param_df_shuffled = pd.DataFrame(
                ctf_slopes_shuffled[param], columns=t_arr
            )
            one_param_df_shuffled["Parameter"] = param
            one_param_df_shuffled = one_param_df_shuffled.melt(
                id_vars=["Parameter"],
                var_name="Time (s)",
                value_name="CTF slope",
            )
            one_param_df_shuffled["Shuffled location labels?"] = "Yes"
            one_param_df = pd.concat(
                (one_param_df, one_param_df_shuffled), axis=0
            )
        ctf_slopes_dfs.append(one_param_df)

    # Combine DataFrames of CTF slopes for each parameter into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()

    # Plot CTF slope time course for each parameter
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    ctf_slopes_big_df["CTF slope"] = -ctf_slopes_big_df["CTF slope"]
    ci = 95 if plot_errorbars else None
    ax = sns.lineplot(
        data=ctf_slopes_big_df,
        hue="Parameter",
        x="Time (s)",
        y="CTF slope",
        style="Shuffled location labels?",
        palette=palette,
        legend="brief",
        ci=ci,
        ax=ax,
    )

    # Plot aesthetics
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
    if task_num != 0 or not save_fname:
        legend.remove()
    _, _, _, ymax = ax.axis()
    if plot_timings:
        ax.axvline(0.0, c="gray", ls="--")
        ax.text(0.03, ymax, "Stimulus onset", va="bottom", ha="right", size=24)
        ax.axvline(task_timings[0], c="gray", ls="--")
        offset_x, offset_ha = task_timings[0] + 0.03, "left"
        if task_timings[0] > 0.75:
            offset_x, offset_ha = task_timings[0], "center"
        ax.text(
            offset_x,
            ymax,
            "Stimulus offset",
            va="bottom",
            ha=offset_ha,
            size=24,
        )
        ax.axvline(task_timings[1], c="gray", ls="--")
        ax.text(
            task_timings[1],
            ymax,
            "Free response",
            va="bottom",
            ha="center",
            size=24,
        )
    ax.set_title(
        f"Task {task_num + 1} (n = {n})",
        fontsize=48,
        fontweight="bold",
        y=1.08,
    )
    ax.set_xlabel("Time (s)", size=28)
    ax.set_ylabel("CTF slope", size=28)
    ax.tick_params(labelsize=20)
    sns.despine(ax=ax)

    # Save if desired
    if save_fname:
        plt.savefig(save_fname, bbox_inches="tight", dpi=300)
        plt.close()


def plot_ctf_slope_paired_ttest(
    ctf_slopes,
    t_arrays,
    t_window,
    ctf_slopes_shuffled=None,
    palette=None,
    save_fname=None,
    task_timings=params.TASK_TIMINGS,
):
    """Plot paired t-tests of channel tuning function (CTF) slope averaged
    time window for multiple parameters.
    """
    # Make empty list for CTF slope DataFrames
    ctf_slopes_dfs = []
    for i, (ctf_slope, t_arr) in enumerate(zip(ctf_slopes, t_arrays)):
        # If time window is None, use task timings
        if t_window == "delay":
            t_window = task_timings[i]

        # Average across time window
        t_window_idx = np.where(
            (t_arr >= t_window[0]) & (t_arr <= t_window[1])
        )[0]
        ctf_slope = -np.mean(ctf_slope[:, t_window_idx], axis=1)
        if ctf_slopes_shuffled is not None:
            ctf_slope_shuffled = -np.mean(
                ctf_slopes_shuffled[i][:, t_window_idx], axis=1
            )

        # Make DataFrame of CTF slopes
        ctf_slope_df = pd.DataFrame(
            ctf_slope, columns=["CTF slope"]
        ).reset_index(names="trial")
        ctf_slope_df.loc[:, "Shuffled location labels?"] = "No"
        ctf_slope_df.loc[:, "Task"] = str(i + 1)

        # Add DataFrame of CTF slopes for shuffled location labels if desired
        if ctf_slope_shuffled is not None:
            ctf_slope_shuffled_df = pd.DataFrame(
                ctf_slope_shuffled, columns=["CTF slope"]
            ).reset_index(names="trial")
            ctf_slope_shuffled_df.loc[:, "Shuffled location labels?"] = "Yes"
            ctf_slope_shuffled_df.loc[:, "Task"] = str(i + 1)
            ctf_slope_df = pd.concat(
                (ctf_slope_df, ctf_slope_shuffled_df), axis=0
            )

        # Add DataFrame to list
        ctf_slopes_dfs.append(ctf_slope_df)

    # Combine DataFrames of CTF slopes from each task into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()

    # Plot paired t-tests of CTF slopes for each task
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=ctf_slopes_big_df,
        x="Task",
        y="CTF slope",
        split=True,
        hue="Shuffled location labels?",
        palette=palette,
        inner="stick",
    )
    pairs = [
        ((str(task_num), "No"), (str(task_num), "Yes"))
        for task_num in list(set(ctf_slopes_big_df["Task"]))
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=ctf_slopes_big_df,
        x="Task",
        y="CTF slope",
        hue="Shuffled location labels?",
        plot="violinplot",
    )
    annotator.configure(
        test="t-test_paired",
        text_format="star",
        comparisons_correction="bonferroni",
    )
    annotator.apply_and_annotate()
    sns.despine(ax=ax)

    # Save figure
    if save_fname:
        plt.savefig(save_fname, bbox_inches="tight", dpi=300)
    return


def train_and_test_one_subj(
    subj,
    param,
    param_dir,
    trial_split_criterion=None,
    n_blocks=params.N_BLOCKS,
    n_block_iters=params.N_BLOCK_ITERS,
    output_dir=params.IEM_OUTPUT_DIR,
    verbose=True,
):
    """Train and test one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to use for decoding.

    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to use for cross-validation.
    n_block_iters : int (default: params.N_BLOCK_ITERS)
        Number of iterations to use for cross-validation.
    save_dir : str (default: params.IEM_OUTPUT_DIR)
        Directory to save output to.

    Returns
    -------
    ctf_slope : np.ndarray
        Array containing CTF slope across time.
    times : np.ndarray
        Array containing time points.
    """
    # Start timer
    start = time.time()

    # Load parameterized data
    epochs, times, beh_data, param_data = load_param_data(
        subj,
        param,
        param_dir,
    )

    # Split trials based on selection criteria
    beh_data_set = (beh_data,)
    param_data_set = (param_data,)
    tags = ("",)
    if trial_split_criterion is not None:
        tags = ("high", "low")
        trials_high, trials_low = split_trials(
            subj, param_dir, **trial_split_criterion
        )
        beh_data_set = (
            beh_data[trials_high],
            beh_data[trials_low],
        )
        param_data_set = (
            param_data[trials_high],
            param_data[trials_low],
        )
        split_param = trial_split_criterion["param"]
        split_t_window = trial_split_criterion["t_window"]
        split_baseline = None
        if "baseline_t_window" in trial_split_criterion:
            split_baseline = trial_split_criterion["baseline_t_window"]
        output_dir = os.path.join(
            output_dir, f"{split_param}_{split_t_window}"
        )
        if split_baseline is not None:
            operation = "diff"
            if "operation" in trial_split_criterion:
                operation = trial_split_criterion["operation"]
            output_dir = f"{output_dir}_{operation}_{split_baseline}"

    for tag, beh_data, param_data in zip(tags, beh_data_set, param_data_set):
        # Make directories specific to parameter and trial split (if there is one)
        save_dir = os.path.join(output_dir, param, tag)
        os.makedirs(save_dir, exist_ok=True)

        # Load data if already done
        ctf_slope_fname = f"{save_dir}/ctf_slope_{subj}.npy"
        ctf_slope_null_fname = f"{save_dir}/ctf_slope_null_{subj}.npy"

        if os.path.exists(ctf_slope_fname) and os.path.exists(
            ctf_slope_null_fname
        ):
            # Load slope array
            mean_ctf_slope = np.load(ctf_slope_fname)

            # Load null slope array
            mean_ctf_slope_null = np.load(ctf_slope_null_fname)

            # Print loading time if desired
            if verbose:
                print(
                    f"Loading {param} data for {subj} took "
                    f"{time.time() - start:.3f} s"
                )
            return mean_ctf_slope, mean_ctf_slope_null, times

        # Iterate through sets of blocks
        n_timepts = len(times)
        ctf_slope = np.zeros((n_block_iters, n_blocks, n_timepts))
        ctf_slope_null = np.zeros((n_block_iters, n_blocks, n_timepts))
        for block_iter in range(n_block_iters):
            # Average parameterized data within trial blocks
            param_arr = average_param_data_within_trial_blocks(
                epochs, times, beh_data, param_data
            )

            # Iterate through blocks
            for test_block_num in range(n_blocks):
                # Split into training and testing data
                train_data = np.delete(
                    param_arr, test_block_num, axis=1
                ).reshape(param_arr.shape[0], -1, param_arr.shape[-1])
                test_data = param_arr[:, test_block_num, :, :]

                # Create labels for training and testing
                train_labels = np.tile(IEM().channel_centers, 2)
                test_labels = IEM().channel_centers

                # Train IEMs for block of data, with no shuffle
                ctf_slope[block_iter, test_block_num, :] = iem_one_block(
                    train_data, train_labels, test_data, test_labels
                )

                # Train IEMs for block of data, with shuffle
                train_labels_shuffled = np.random.permutation(train_labels)
                ctf_slope_null[block_iter, test_block_num, :] = iem_one_block(
                    train_data, train_labels_shuffled, test_data, test_labels
                )

        # Average across blocks and block iterations
        mean_ctf_slope = np.mean(ctf_slope, axis=(0, 1))
        mean_ctf_slope_null = np.mean(ctf_slope_null, axis=(0, 1))

        # Save data to avoid unnecessary re-processing
        np.save(ctf_slope_fname, mean_ctf_slope)
        np.save(ctf_slope_null_fname, mean_ctf_slope_null)

    # Print processing time if desired
    if verbose:
        print(
            f"Processing {param} data for {subj} took "
            f"{time.time() - start:.3f} s"
        )
    return mean_ctf_slope, mean_ctf_slope_null, times


def train_and_test_all_subjs(
    param,
    param_dir,
    task_num=None,
    trial_split_criterion=None,
    subjects_by_task=params.SUBJECTS_BY_TASK,
    output_dir=params.IEM_OUTPUT_DIR,
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
    mean_ctf_slopes : np.ndarray
        Array containing CTF slopes across subjects.
    t_arr : np.ndarray
        Array containing time points.
    """
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
    if task_num is not None:
        subjects_by_task = [subjects_by_task[task_num]]
    subjs = []
    for experiment, subj_ids in subjects_by_task:
        subjs.extend(
            ["_".join((experiment, str(subj_id))) for subj_id in subj_ids]
        )
    subjs = sorted(list(set(subjs) & set(subjs_processed)))

    # Initialize arrays to store data across subjects by experiment
    mean_ctf_slopes = [[] for _ in range(len(subjects_by_task))]
    mean_ctf_slopes_null = [[] for _ in range(len(subjects_by_task))]
    t_arrays = [[] for _ in range(len(subjects_by_task))]

    # Process each subject's data
    for subj in subjs:
        # Train and test for one subject
        (
            mean_ctf_slope,
            mean_ctf_slope_null,
            t_arr,
        ) = train_and_test_one_subj(
            subj,
            param,
            param_dir,
            trial_split_criterion=trial_split_criterion,
            output_dir=output_dir,
        )

        # Add data to big arrays
        experiment, subj_num = subj.split("_")
        task_num = np.argmax(
            [
                exp == experiment and int(subj_num) in ids
                for exp, ids in subjects_by_task
            ]
        )
        mean_ctf_slopes[task_num].append(mean_ctf_slope)
        mean_ctf_slopes_null[task_num].append(mean_ctf_slope_null)
        t_arrays[task_num] = t_arr

    # Collate CTF slopes across subjects
    mean_ctf_slopes = [np.array(ctf_slope) for ctf_slope in mean_ctf_slopes]
    mean_ctf_slopes_null = [
        np.array(ctf_slope_null) for ctf_slope_null in mean_ctf_slopes_null
    ]

    return (
        mean_ctf_slopes,
        mean_ctf_slopes_null,
        t_arrays,
    )
