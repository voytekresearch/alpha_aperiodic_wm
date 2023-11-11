"""Train and test IEM using the same methodology used by Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/)."""

# Import neccesary modules
import os.path
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne
import params
from iem import IEM


def load_param_data(
        subj, param, param_dir, threshold_param=None, threshold_val=None,
        processed_dir=params.PROCESSED_DIR, sparam_dir=params.SPARAM_DIR,
        decim_factor=params.DECIM_FACTOR):
    """Load processed EEG and behavioral data for one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to decode.
    threshold_param : str | None (default: None)
        Parameter to threshold.
    threshold_val : float | None (default: None)
        Threshold value.
    processed_dir : str (default: params.PROCESSED_DIR)
        Path to directory containing processed data.
    decim_factor : int (default: params.DECIM_FACTOR)
        Decimation factor.

    Returns
    -------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : ndarray
        Time course.
    beh_data : dict
        Dictionary containing behavioral data.
    param_data : ndarray
        Array containing parameterized data for decoding.
    """
    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f'{subj}_beh.npy'))
    beh_nan = np.isnan(beh_data)

    # Load any skipped trials
    skipped_fname = f'{sparam_dir}/skipped.csv'
    if os.path.exists(skipped_fname):
        skipped_df = pd.read_csv(skipped_fname)
        skipped_dct = skipped_df.to_dict(orient='list')
        if subj in skipped_dct.keys():
            beh_nan[skipped_dct[subj]] = True

    # Remove trials with NaNs
    beh_data = beh_data[~beh_nan]

    # Load epoched EEG data
    epochs = mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_eeg_epo.fif'), preload=True, verbose=False)
    epochs.drop(beh_nan, verbose=False)

    # Get times from epochs, taking account of decimation if applied
    times = epochs.times
    if param != 'total_power':
        times = epochs.times[::decim_factor]

    # Load parameterized data
    param_data = mne.read_epochs(os.path.join(
        param_dir, f'{subj}_{param}_epo.fif'), verbose=False).get_data()
    param_data = param_data[~beh_nan, :, :]

    # Load parameterized data for threhsold parameter
    if threshold_param is not None and threshold_val is not None:
        thresh_fname = os.path.join(
            param_dir, f'{subj}_{threshold_param}_epo.fif')
        thresh_data =  mne.read_epochs(thresh_fname, verbose=False).get_data()
        thresh_data = thresh_data[~beh_nan, :, :]
        param_data = param_data[thresh_data >= threshold_val, :, :]
        beh_data = beh_data[thresh_data >= threshold_val]
        epochs.drop(thresh_data < threshold_val, verbose=False)
    return epochs, times, beh_data, param_data


def average_param_data_within_trial_blocks(
        epochs, times, beh_data, param_data, n_blocks=params.N_BLOCKS):
    """Averaging parameterized data across trials within a block for each
    location bin.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : ndarray
        Time course.
    beh_data : dict
        Dictionary containing behavioral data.
    param_data : ndarray
        Array containing parameterized data for decoding.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to split trials into.

    Returns
    -------
    param_arr : ndarray
        Array containing averaged parameterized data for decoding.
    """
    # Extract relative variables from data
    assert np.count_nonzero(np.isnan(beh_data)) == 0
    n_bins = np.sum(~np.isnan(np.unique(beh_data)))
    n_channels = epochs.get_data().shape[-2]
    n_timepts = len(times)

    # Determine number of trials per location bin
    _, counts = np.unique(beh_data, return_counts=True)
    n_trials_per_bin = counts.min() // n_blocks * n_blocks
    idx_split_by_vals = np.split(np.argsort(beh_data), np.where(np.diff(sorted(
            beh_data)))[0]+1)

    # Calculate parameterized data for block of trials
    param_arr = np.zeros((n_blocks, n_bins, n_channels, n_timepts))

    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Average parameterized data across block of trials
        block_avg = np.real(np.nanmean(param_data[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts), axis=0))

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
    train_data : ndarray
        Array containing training data.
    train_labels : ndarray
        Array containing training labels.
    test_data : ndarray
        Array containing testing data.
    test_labels : ndarray
        Array containing testing labels.

    Returns
    -------
    mean_channel_offset : ndarray
        Array containing mean channel offset from fitted IEM.
    ctf_slope : ndarray
        Array containing CTF slope from fitted IEM.
    """
    # Initialize IEM instance
    iem = IEM()

    # Train IEM using training data
    iem.train_model(train_data, train_labels)

    # Test IEM using testing data
    iem.estimate_ctf(test_data, test_labels)

    # Compute mean channel offset and CTF slope
    iem.compute_mean_channel_offset()
    iem.compute_ctf_slope()
    return iem.mean_channel_offset, iem.ctf_slope


def iem_one_block(
        train_data, train_labels, test_data, test_labels, time_axis=-1):
    """Estimate channel response function (CRF) for one block of training and
    testing data across all time points.

    Parameters
    ----------
    train_data : ndarray
        Array containing training data.
    train_labels : ndarray
        Array containing training labels.
    test_data : ndarray
        Array containing testing data.
    test_labels : ndarray
        Array containing testing labels.
    time_axis : int (default: -1)
        Axis of data array containing time points.  Default is -1, which
        corresponds to the last axis.

    Returns
    -------
    mean_channel_offset : ndarray
        Array containing mean channel offset from fitted IEM.
    ctf_slope : ndarray
        Array containing CTF slope from fitted IEM.
    """
    # Organize data into lists of arrays for each time point
    data_one_tp = [list(np.rollaxis(
        data, time_axis)) for data in (train_data, test_data)]

    # Make arguments for multiprocessing such that each function call will
    # contain training and testing data and labels for one time point
    args = tuple([(
        train_data_one_tp, train_labels, test_data_one_tp, test_labels) for
            train_data_one_tp, test_data_one_tp in zip(*data_one_tp)])

    # Parallelize training and testing of IEM across time points
    with mp.Pool() as pool:
        pool_out = pool.starmap(iem_one_timepoint, args)
    mean_channel_offset, ctf_slope = list(zip(*pool_out))
    return np.array(mean_channel_offset).T, np.array(ctf_slope)


def plot_channel_offset(channel_offset_arr, t_arr, save_fname=None):
    """Plot channel offset across time.

    Parameters
    ----------
    channel_offset_arr : ndarray
        Array containing channel offset.
    t_arr : ndarray
        Array containing time points.
    save_fname : str (default: None)
        File name to save figure to.  If None, figure will not be saved.
    """
    # Initialize figure
    plt.figure()

    # Plot channel offset
    offset_range = channel_offset_arr.shape[0]
    offset_vals = np.linspace(
        -offset_range // 2, offset_range // 2, num=offset_range).astype(int)
    plt.pcolormesh(
        t_arr, offset_vals, channel_offset_arr, cmap='YlGnBu_r', shading='auto')
    plt.gca().invert_yaxis()

    # Label axes
    plt.xlabel('Time (s)')
    plt.ylabel('Channel Offset')

    # Add colorbar
    plt.colorbar(label='Channel Response')

    # Save if desired
    if save_fname:
        plt.savefig(save_fname)
    plt.close()


def plot_ctf_slope(
        ctf_slopes, t_arr, task_num, task_timings, ctf_slopes_shuffled=None,
        palette=None, save_fname=None):
    """Plot channel tuning function (CTF) across time for multiple
    parameters.

    Parameters
    ----------
    ctf_slopes : dict
        Dictionary containing CTF slopes for each parameter. Keys are parameter
        names and values are arrays containing CTF slopes.
    t_arr : ndarray
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
        one_param_df['Parameter'] = param
        one_param_df = one_param_df.melt(
            id_vars=['Parameter'], var_name='Time (s)', value_name='CTF slope')
        one_param_df['Shuffled location labels'] = 'No'

        # Add DataFrame of CTF slopes for shuffled location labels if desired
        if ctf_slopes_shuffled is not None:
            one_param_df_shuffled = pd.DataFrame(
                ctf_slopes_shuffled[param], columns=t_arr)
            one_param_df_shuffled['Parameter'] = param
            one_param_df_shuffled = one_param_df_shuffled.melt(
                id_vars=['Parameter'], var_name='Time (s)',
                value_name='CTF slope')
            one_param_df_shuffled['Shuffled location labels'] = 'Yes'
            one_param_df = pd.concat(
                (one_param_df, one_param_df_shuffled), axis=0)
        ctf_slopes_dfs.append(one_param_df)

    # Combine DataFrames of CTF slopes for each parameter into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()
    ctf_slopes_big_df.to_csv('ctf_slopes.csv')

    # Plot CTF slope time course for each parameter
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(
        data=ctf_slopes_big_df, hue='Parameter', x='Time (s)', y='CTF slope',
        style='Shuffled location labels', palette=palette, legend='brief')
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1), ncol=2)
    if task_num > 0:
        legend.remove()
    _, _, _, ymax = plt.axis()
    plt.axvline(0.0, c='gray', ls='--')
    plt.text(0.03, ymax, 'Stimulus onset', va='bottom', ha='right', size=16)
    plt.axvline(task_timings[0], c='gray', ls='--')
    offset_x, offset_ha = task_timings[0] + 0.03, 'left'
    if task_timings[0] > 0.75:
        offset_x, offset_ha = task_timings[0], 'center'
    plt.text(
        offset_x, ymax, 'Stimulus offset', va='bottom', ha=offset_ha, size=16)
    plt.axvline(task_timings[1], c='gray', ls='--')
    plt.text(
        task_timings[1], ymax, 'Free response', va='bottom', ha='center',
        size=16)
    plt.title(
        f'Task {task_num + 1} (n = {n})', size=28, y=1.08, fontweight='bold')
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('CTF slope', size=20)
    plt.xticks(size=12)
    plt.yticks(size=12)
    sns.despine()

    # Save if desired
    if save_fname:
        plt.savefig(save_fname, bbox_inches='tight', dpi=300)
    plt.close()


def train_and_test_one_subj(
        subj, param, param_dir, threshold_param=None, threshold_val=None,
        n_blocks=params.N_BLOCKS, n_block_iters=params.N_BLOCK_ITERS,
        save_dir=params.IEM_OUTPUT_DIR, fig_dir=params.FIG_DIR, verbose=True):
    """Train and test one subject.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to use for decoding.
    threshold_param : str (default: None)
        Parameter to use for thresholding.  If None, no thresholding will be
        done.
    threshold_val : float (default: None)
        Value to use for thresholding.  If None, no thresholding will be done.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to use for cross-validation.
    n_block_iters : int (default: params.N_BLOCK_ITERS)
        Number of iterations to use for cross-validation.
    save_dir : str (default: params.IEM_OUTPUT_DIR)
        Directory to save output to.
    fig_dir : str (default: params.FIG_DIR)
        Directory to save figures to.

    Returns
    -------
    mean_channel_offset : ndarray
        Array containing mean channel offset across time.
    ctf_slope : ndarray
        Array containing CTF slope across time.
    times : ndarray
        Array containing time points.
    """
    # Start timer
    start = time.time()

    # Make directories specific to parameter
    save_dir = os.path.join(save_dir, param)
    fig_dir = os.path.join(fig_dir, param)
    if threshold_param is not None and threshold_val is not None:
        save_dir = os.path.join(save_dir, f'{threshold_param}>{threshold_val}')
        fig_dir = os.path.join(fig_dir, f'{threshold_param}>{threshold_val}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load parameterized data
    epochs, times, beh_data, param_data = load_param_data(
        subj, param, param_dir, threshold_param=threshold_param,
        threshold_val=threshold_val)

    # Load channel offset data if already done
    channel_offset_fname = f'{save_dir}/channel_offset_{subj}.npy'
    ctf_slope_fname = f'{save_dir}/ctf_slope_{subj}.npy'
    ctf_slope_null_fname = f'{save_dir}/ctf_slope_null_{subj}.npy'

    if os.path.exists(channel_offset_fname) and os.path.exists(
            ctf_slope_fname) and os.path.exists(ctf_slope_null_fname):
        # Load offset array
        mean_channel_offset = np.load(channel_offset_fname)

        # Load slope array
        mean_ctf_slope = np.load(ctf_slope_fname)

        # Load null slope array
        mean_ctf_slope_null = np.load(ctf_slope_null_fname)

        # Plot channel offset and save
        fig_fname = os.path.join(fig_dir, f'channel_offset_{subj}')
        plot_channel_offset(
            mean_channel_offset, times, save_fname=fig_fname)

        # Print loading time if desired
        if verbose:
            print(f'Loading {param} data for {subj} took '
                  f'{time.time() - start:.3f} s')
        return mean_channel_offset, mean_ctf_slope, mean_ctf_slope_null, times

    # Iterate through sets of blocks
    n_timepts = len(times)
    channel_offsets = np.zeros((
        n_block_iters, n_blocks, IEM().feat_space_range, n_timepts))
    ctf_slope = np.zeros((n_block_iters, n_blocks, n_timepts))
    ctf_slope_null = np.zeros((n_block_iters, n_blocks, n_timepts))
    for block_iter in range(n_block_iters):
        # Average parameterized data within trial blocks
        param_arr = average_param_data_within_trial_blocks(
            epochs, times, beh_data, param_data)

        # Iterate through blocks
        for test_block_num in range(n_blocks):
            # Split into training and testing data
            train_data = np.delete(
                param_arr, test_block_num, axis=1).reshape(
                    param_arr.shape[0], -1, param_arr.shape[-1])
            test_data = param_arr[:, test_block_num, :, :]

            # Create labels for training and testing
            train_labels = np.tile(IEM().channel_centers, 2)
            test_labels = IEM().channel_centers

            # Train IEMs for block of data, with no shuffle
            channel_offsets[block_iter, test_block_num, :, :], \
                ctf_slope[block_iter, test_block_num, :] = iem_one_block(
                    train_data, train_labels, test_data, test_labels)

            # Train IEMs for block of data, with shuffle
            train_labels_shuffled = np.random.permutation(train_labels)
            _, ctf_slope_null[block_iter, test_block_num, :] = iem_one_block(
                    train_data, train_labels_shuffled, test_data, test_labels)

    # Average across blocks and block iterations
    mean_channel_offset = np.mean(channel_offsets, axis=(0, 1))
    mean_ctf_slope = np.mean(ctf_slope, axis=(0, 1))
    mean_ctf_slope_null = np.mean(ctf_slope_null, axis=(0, 1))

    # Save data to avoid unnecessary re-processing
    np.save(channel_offset_fname, mean_channel_offset)
    np.save(ctf_slope_fname, mean_ctf_slope)
    np.save(ctf_slope_null_fname, mean_ctf_slope_null)

    # Plot channel offset and save
    fig_fname = os.path.join(fig_dir, f'channel_offset_{subj}')
    plot_channel_offset(
        mean_channel_offset, times, save_fname=fig_fname)

    # Print processing time if desired
    if verbose:
        print(f'Processing {param} data for {subj} took '
              f'{time.time() - start:.3f} s')
    return mean_channel_offset, mean_ctf_slope, mean_ctf_slope_null, times


def train_and_test_all_subjs(
        param, param_dir, task_num=None, threshold_param=None,
        threshold_val=None, fig_dir=params.FIG_DIR,
        subjects_by_task=params.SUBJECTS_BY_TASK):
    """Train and test for all subjects,.

    Parameters
    ----------
    param : str
        Parameter to use for decoding.
    param_dir : str
        Directory containing parameterized data.
    threshold_param : str (default: None)
        Parameter to use for thresholding.  If None, no thresholding will be
        done.
    threshold_val : float (default: None)
        Value to use for thresholding.  If None, no thresholding will be done.
    fig_dir : str (default: params.FIG_DIR)
        Directory to save figures to.

    Returns
    -------
    mean_channel_offset_all_subjs : ndarray
        Array containing mean channel offset across subjects.
    mean_ctf_slopes : ndarray
        Array containing CTF slopes across subjects.
    t_arr : ndarray
        Array containing time points.
    """
    # Get all subject IDs
    subjs = sorted(['_'.join(f.split('_')[:2]) for f in os.listdir(
        param_dir) if param in f])

    # If desired, only use subjects from one task
    if task_num is not None:
        experiment, subj_ids = subjects_by_task[task_num]
        subjs = ['_'.join((experiment, subj_id)) for subj_id in subj_ids]
        subjects_by_task = [subjects_by_task[task_num]]

    # Initialize arrays to store data across subjects by experiment
    mean_channel_offsets = [[] for _ in range(len(subjects_by_task))]
    mean_ctf_slopes = [[] for _ in range(len(subjects_by_task))]
    mean_ctf_slopes_null = [[] for _ in range(len(subjects_by_task))]
    t_arrays = [[] for _ in range(len(subjects_by_task))]

    # Process each subject's data
    for subj in subjs:
        # Train and test for one subject
        mean_channel_offset, mean_ctf_slope, mean_ctf_slope_null, t_arr = \
            train_and_test_one_subj(
                subj, param, param_dir, threshold_param=threshold_param,
                threshold_val=threshold_val)

        # Add data to big arrays
        experiment, subj_num = subj.split('_')
        task_num = np.argmax([exp == experiment and int(
            subj_num) in ids for exp, ids in subjects_by_task])
        mean_channel_offsets[task_num].append(mean_channel_offset)
        mean_ctf_slopes[task_num].append(mean_ctf_slope)
        mean_ctf_slopes_null[task_num].append(mean_ctf_slope_null)
        t_arrays[task_num] = t_arr

    # Combine channel offsets across subjects
    mean_channel_offset_all_subjs = [np.mean(
        channel_offset, axis=0) for channel_offset in mean_channel_offsets]

    # Collate CTF slopes across subjects
    mean_ctf_slopes = [np.array(ctf_slope) for ctf_slope in mean_ctf_slopes]
    mean_ctf_slopes_null = [np.array(
        ctf_slope_null) for ctf_slope_null in mean_ctf_slopes_null]

    # Plot channel offset across subjects
    fig_dir = os.path.join(fig_dir, param)
    if threshold_param is not None and threshold_val is not None:
        fig_dir = os.path.join(fig_dir, f'{threshold_param}>{threshold_val}')
    for i, (experiment, _) in enumerate(subjects_by_task):
        fig_fname = os.path.join(
            fig_dir, f'channel_offset_{experiment}_task{i}')
        plot_channel_offset(
            mean_channel_offset_all_subjs[i], t_arrays[i], save_fname=fig_fname)
    return mean_channel_offsets, mean_ctf_slopes, mean_ctf_slopes_null, t_arrays
