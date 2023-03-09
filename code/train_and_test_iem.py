"""Train and test IEM using the same methodology used by Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/)."""

# Import neccesary modules
import os.path
import multiprocessing as mp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne
import params
from iem import IEM


def load_processed_data(
        subj, param, threshold_param=None, threshold_val=None,
        processed_dir=params.PROCESSED_DIR, decim_factor=params.DECIM_FACTOR):
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
    processed_data : ndarray
        Array containing processed data for decoding.
    """
    # Load epoched EEG data
    epochs = mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_eeg_data_epo.fif'), preload=True)

    # Get times from epochs, taking account of decimation if applied
    times = epochs.times
    if param != 'total_power':
        times = epochs.times[::decim_factor]

    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f'{subj}_beh_data.npz'))

    # Load processed data
    processed_data = mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_{param}_epo.fif')).get_data()

    # Load processed data for threhsold parameter
    if threshold_param is not None and threshold_val is not None:
        thresh_data =  mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_{threshold_param}_epo.fif')).get_data()
        processed_data[thresh_data < threshold_val] = np.nan
    return epochs, times, beh_data, processed_data


def average_processed_data_within_trial_blocks(
        epochs, times, beh_data, processed_data, n_blocks=params.N_BLOCKS):
    """Averaging processed data across trials within a block for each
    location bin.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object containing EEG data.
    times : ndarray
        Time course.
    beh_data : dict
        Dictionary containing behavioral data.
    processed_data : ndarray
        Array containing processed data for decoding.
    n_blocks : int (default: params.N_BLOCKS)
        Number of blocks to split trials into.

    Returns
    -------
    processed_arr : ndarray
        Array containing averaged processed data for decoding.
    """
    # Extract relative variables from data
    pos_bins = beh_data['posBin']
    n_bins = len(set(pos_bins))
    n_channels = epochs.get_data().shape[-2]
    n_timepts = len(times)

    # Determine number of trials per location bin
    n_trials_per_bin = np.bincount(pos_bins - 1).min() // n_blocks * n_blocks
    idx_split_by_vals = np.split(np.argsort(pos_bins), np.where(np.diff(sorted(
            pos_bins)))[0]+1)

    # Calculate processed data for block of trials
    processed_arr = np.zeros((n_blocks, n_bins, n_channels, n_timepts))

    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Average processed data across block of trials
        block_avg = np.real(np.nanmean(processed_data[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts), axis=0))

        # Add block-averaged processed data to array
        processed_arr[:, i, :, :] = block_avg

    # Rearrange axes of data to work for IEM pipeline
    processed_arr = np.moveaxis(processed_arr, 2, 0)
    return processed_arr


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


def plot_ctf_slope(ctf_slopes, t_arr, palette=None, save_fname=None):
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
        one_param_df = pd.DataFrame(ctf_slopes_one_param, columns=t_arr)
        one_param_df['Parameter'] = param
        one_param_df = one_param_df.melt(
            id_vars=['Parameter'], var_name='Time (s)', value_name='CTF slope')
        ctf_slopes_dfs.append(one_param_df)

    # Combine DataFrames of CTF slopes for each parameter into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()

    # Plot CTF slope time course for each parameter
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=ctf_slopes_big_df, hue='Parameter', x='Time (s)', y='CTF slope',
        palette=palette)
    plt.axvline(0.0, c='gray', ls='--')
    plt.axvline(0.25, c='gray', ls='--')
    plt.axvline(2.0, c='gray', ls='--')
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('CTF slope', size=20)
    plt.xticks(size=12)
    plt.yticks(size=12)
    sns.despine()

    # Save if desired
    if save_fname:
        plt.savefig(save_fname)
    plt.close()


def train_and_test_one_subj(
        subj, param, threshold_param=None, threshold_val=None,
        n_blocks=params.N_BLOCKS, n_block_iters=params.N_BLOCK_ITERS,
        save_dir=params.IEM_OUTPUT_DIR, fig_dir=params.FIG_DIR):
    """Reproduce one subject.

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
    # Make directories specific to parameter
    save_dir = os.path.join(save_dir, param)
    fig_dir = os.path.join(fig_dir, param)
    if threshold_param is not None and threshold_val is not None:
        save_dir = os.path.join(save_dir, f'{threshold_param}>{threshold_val}')
        fig_dir = os.path.join(fig_dir, f'{threshold_param}>{threshold_val}')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Load processed data
    epochs, times, beh_data, processed_data = load_processed_data(
        subj, param, threshold_param=threshold_param,
        threshold_val=threshold_val)

    # Load channel offset data if already done
    channel_offset_fname = os.path.join(save_dir, f'channel_offset_{subj}.npy')
    ctf_slope_fname = os.path.join(save_dir, f'ctf_slope_{subj}.npy')
    if os.path.exists(channel_offset_fname) and os.path.exists(
            ctf_slope_fname):
        # Load offset array
        mean_channel_offset = np.load(channel_offset_fname)

        # Load slope array
        mean_ctf_slope = np.load(ctf_slope_fname)

        # Plot channel offset and save
        fig_fname = os.path.join(fig_dir, f'channel_offset_{subj}')
        plot_channel_offset(
            mean_channel_offset, times, save_fname=fig_fname)
        return mean_channel_offset, mean_ctf_slope, times

    # Iterate through sets of blocks
    n_timepts = len(times)
    channel_offsets = np.zeros((
        n_block_iters, n_blocks, IEM().feat_space_range, n_timepts))
    ctf_slope = np.zeros((n_block_iters, n_blocks, n_timepts))
    for block_iter in range(n_block_iters):
        # Average processed data within trial blocks
        processed_arr = average_processed_data_within_trial_blocks(
            epochs, times, beh_data, processed_data)

        # Iterate through blocks
        for test_block_num in range(n_blocks):
            # Split into training and testing data
            train_data = np.delete(
                processed_arr, test_block_num, axis=1).reshape(
                    processed_arr.shape[0], -1, processed_arr.shape[-1])
            test_data = processed_arr[:, test_block_num, :, :]

            # Create labels for training and testing
            train_labels = np.tile(IEM().channel_centers, 2)
            test_labels = IEM().channel_centers

            # Train IEMs for block of data
            channel_offsets[block_iter, test_block_num, :, :], \
                ctf_slope[block_iter, test_block_num, :] = iem_one_block(
                    train_data, train_labels, test_data, test_labels)

    # Average across blocks and block iterations
    mean_channel_offset = np.mean(channel_offsets, axis=(0, 1))
    mean_ctf_slope = np.mean(ctf_slope, axis=(0, 1))

    # Save data to avoid unnecessary re-processing
    np.save(channel_offset_fname, mean_channel_offset)
    np.save(ctf_slope_fname, mean_ctf_slope)

    # Plot channel offset and save
    fig_fname = os.path.join(fig_dir, f'channel_offset_{subj}')
    plot_channel_offset(
        mean_channel_offset, times, save_fname=fig_fname)
    return mean_channel_offset, mean_ctf_slope, times


def train_and_test_all_subjs(
        param, threshold_param=None, threshold_val=None,
        processed_dir=params.PROCESSED_DIR, fig_dir=params.FIG_DIR):
    """Reproduce all subjects.

    Parameters
    ----------
    param : str
        Parameter to use for decoding.
    threshold_param : str (default: None)
        Parameter to use for thresholding.  If None, no thresholding will be
        done.
    threshold_val : float (default: None)
        Value to use for thresholding.  If None, no thresholding will be done.
    processed_dir : str (default: params.PROCESSED_DIR)
        Directory containing processed data.
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
    subjs = sorted([f.split('_')[0] for f in os.listdir(
        processed_dir) if param in f])

    # Process each subject's data
    mean_channel_offsets, mean_ctf_slopes = [], []
    for subj in subjs:
        subj_mean_channel_offset, subj_mean_ctf_slope, \
            t_arr = train_and_test_one_subj(
                subj, param, threshold_param=threshold_param,
                threshold_val=threshold_val)
        mean_channel_offsets.append(subj_mean_channel_offset)
        mean_ctf_slopes.append(subj_mean_ctf_slope)

    # Combine channel offsets across subjects
    mean_channel_offset_all_subjs = np.mean(mean_channel_offsets, axis=0)

    # Collate CTF slopes across subjects
    mean_ctf_slopes = np.array(mean_ctf_slopes)

    # Plot channel offset across subjects
    fig_dir = os.path.join(fig_dir, param)
    if threshold_param is not None and threshold_val is not None:
        fig_dir = os.path.join(fig_dir, f'{threshold_param}>{threshold_val}')
    fig_fname = os.path.join(fig_dir, 'channel_offset_all')
    plot_channel_offset(
        mean_channel_offset_all_subjs, t_arr, save_fname=fig_fname)
    return mean_channel_offsets, mean_ctf_slopes, t_arr
