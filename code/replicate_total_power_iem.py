"""Replicate decoding of spatial location from total alpha power using an IEM by
Foster and colleagues (https://pubmed.ncbi.nlm.nih.gov/26467522/)"""

# Import neccesary modules
import os.path
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import mne
import params
from iem import IEM


def load_processed_data(subj, processed_dir=params.PROCESSED_DIR):
    """Load processed EEG and behavioral data for one subject."""
    # Load epoched EEG data
    epochs = mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_eeg_data_epo.fif'), preload=True)

    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f'{subj}_beh_data.npz'))

    # Load alpha total power data
    total_power = mne.read_epochs(os.path.join(
        processed_dir, f'{subj}_total_power_epo.fif'))
    return epochs, beh_data, total_power


def calculate_total_power_for_trial_blocks(
        beh_data, total_power, epochs, n_blocks=params.N_BLOCKS):
    """Calculate total power by averaging across trials within a block for each
    location bin."""
    # Extract relative variables from data
    pos_bins = beh_data['posBin']
    n_bins = len(set(pos_bins))
    n_channels, n_timepts = epochs.get_data().shape[-2:]

    # Determine number of trials per location bin
    n_trials_per_bin = np.bincount(pos_bins - 1).min() // n_blocks * n_blocks
    idx_split_by_vals = np.split(np.argsort(pos_bins), np.where(np.diff(sorted(
            pos_bins)))[0]+1)

    # Calculate total power for block of trials
    total_power_arr = np.zeros((n_blocks, n_bins, n_channels, n_timepts))
    for i, val_split in enumerate(idx_split_by_vals):
        # Randomly permute indices
        idx = np.random.permutation(val_split)[:n_trials_per_bin]

        # Average total power across block of trials
        block_avg = np.real(np.mean(total_power.get_data()[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts), axis=0))

        # Add block-averaged total power to array
        total_power_arr[:, i, :, :] = block_avg

    # Rearrange axes of data to work for IEM pipeline
    total_power_arr = np.moveaxis(total_power_arr, 2, 0)
    return total_power_arr

def iem_one_timepoint(train_data, train_labels, test_data, test_labels):
    """Estimate channel response function (CRF) for one block of training and
    testing data at one time point."""
    # Initialize IEM instance
    iem = IEM()

    # Train IEM using training data
    iem.train_model(train_data, train_labels)

    # Test IEM using testing data
    iem.estimate_crf(test_data, test_labels)
    return iem.mean_channel_offset


def iem_one_block(
        train_data, train_labels, test_data, test_labels, time_axis=-1):
    """Estimate channel response function (CRF) for one block of training and
    testing data across all time points."""
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
        mean_channel_offset = pool.starmap(iem_one_timepoint, args)
    return np.array(mean_channel_offset).T


def plot_channel_offset(
        channel_offset_arr, t_arr, save_fname=None, save_dir=params.FIG_DIR):
    """Plot channel offset across time."""
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
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_fname))


def replicate_one_subj(
        subj, n_blocks=params.N_BLOCKS, n_block_iters=params.N_BLOCK_ITERS,
        save_dir=params.CHANNEL_OFFSETS_DIR):
    """Replicate one subject."""
    # Load processed data
    epochs, beh_data, total_power = load_processed_data(subj)

    # Load channel offset data if already done
    save_fname = os.path.join(save_dir, f'channel_offset_{subj}.npy')
    if os.path.exists(save_fname):
        # Load offset array
        mean_channel_offset = np.load(save_fname)

        # Plot channel offset and save
        plot_channel_offset(
            mean_channel_offset, epochs.times,
            save_fname=f'channel_offset_{subj}')
        return mean_channel_offset, epochs.times

    # Iterate through sets of blocks
    n_timepts = epochs.get_data().shape[-1]
    mean_channel_offset = np.zeros((
        n_block_iters, n_blocks, IEM().feat_space_range, n_timepts))
    for block_iter in range(n_block_iters):
        # Calculate total power
        total_power_arr = calculate_total_power_for_trial_blocks(
            beh_data, total_power, epochs)

        # Iterate through blocks
        for test_block_num in range(n_blocks):
            # Split into training and testing data
            train_data = np.delete(
                total_power_arr, test_block_num, axis=1).reshape(
                    total_power_arr.shape[0], -1, total_power_arr.shape[-1])
            test_data = total_power_arr[:, test_block_num, :, :]

            # Create labels for training and testing
            train_labels = np.tile(IEM().channel_centers, 2)
            test_labels = IEM().channel_centers

            # Train IEMs for block of data
            mean_channel_offset[
                block_iter, test_block_num, :, :] = iem_one_block(
                    train_data, train_labels, test_data, test_labels)

    # Average across blocks and block iterations
    mean_channel_offset = np.mean(mean_channel_offset, axis=(0, 1))

    # Save data to avoid unnecessary re-processing
    os.makedirs(save_dir, exist_ok=True)
    np.save(save_fname, mean_channel_offset)

    # Plot channel offset and save
    plot_channel_offset(
        mean_channel_offset, epochs.times, save_fname=f'channel_offset_{subj}')
    return mean_channel_offset, epochs.times


def replicate_all_subjs(
        processed_dir=params.PROCESSED_DIR):
    """Replicate all subjects."""
    # Get all subject IDs
    subjs = sorted([f.split('_')[0] for f in os.listdir(
        processed_dir) if 'total_power' in f])

    # Process each subject's data
    mean_channel_offsets = []
    for subj in subjs:
        subj_mean_channel_offset, t_arr = replicate_one_subj(subj)
        mean_channel_offsets.append(subj_mean_channel_offset)

    # Combine channel offsets across subjects
    mean_channel_offset_all_subjs = np.mean(mean_channel_offsets, axis=0)

    # Plot channel offset across subjects
    plot_channel_offset(
        mean_channel_offset_all_subjs, t_arr, save_fname='channel_offset_all')


if __name__ == '__main__':
    replicate_all_subjs()
