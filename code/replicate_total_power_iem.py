"""Replicate decoding of spatial location from total alpha power using an IEM by
Foster and colleagues (https://pubmed.ncbi.nlm.nih.gov/26467522/)"""

# Import neccesary modules
import os.path
import numpy as np
import mne
import params


def load_processed_data(subj, processed_dir=params.PROCESSED_DIR):
    """Load processed EEG and behavioral data for one subject."""
    # Load epoched EEG data
    epochs = mne.io.read_raw_fif(os.path.join(
        processed_dir, f'{subj}_eeg_data_epo.fif'))

    # Load behavioral data
    beh_data = np.load(os.path.join(processed_dir, f'{subj}_beh_data.npz'))

    # Load alpha total power data
    total_power = mne.io.read_raw_fif(os.path.join(
        processed_dir, f'{subj}_total_power_epo.fif'))
    return epochs, beh_data, total_power


def split_trials_into_blocks(beh_data, n_blocks=params.N_BLOCKS):
    """Split trials into blocks with even number of trials in each block for
    each location bin."""
    # Seperate into blocks
    blocks = [np.array([]) for _ in range(n_blocks)]
    for val in set(beh_data['posBin'] ):
        # Find indices with particular values and randomly permute these indices
        rand_inds = np.random.permutation(
            np.where(beh_data['posBin'] == val)[0])

        # Split these indices into blocks
        blocks_val = np.array_split(rand_inds, n_blocks)

        # Add blocks of particular value to big blocks structure
        blocks = sorted(blocks, key=len)
        blocks = [np.concatenate((
            block, block_val)) for block, block_val in zip(blocks, blocks_val)]

    # Order blocks by length (max difference between any two blocks' length
    # should be 1)
    blocks = sorted(blocks, key=len)
    return blocks


def calculate_total_power_for_block(
    beh_data, total_power, epochs, n_blocks=params.N_BLOCKS):
    """Calculate total power by averaging across trials within a block for each
    location bin."""
    # Extract relative variables from data
    pos_bins = beh_data['posBin']
    n_channels, n_timepts = epochs.get_data().shape[-2:]

    # Determine number of trials per location bin
    n_trials_per_bin = np.bincount(pos_bins - 1).min() // n_blocks * n_blocks
    idx_split_by_vals = np.split(np.argsort(pos_bins), np.where(np.diff(sorted(
            pos_bins)))[0]+1)

    # Calculate total power for block of trials
    total_power_arr = np.zeros((len(set(
        pos_bins)) * n_blocks, n_channels, n_timepts))
    for i, val_split in enumerate(idx_split_by_vals):
        idx = np.random.permutation(val_split)[:n_trials_per_bin]
        block_avg = np.real(np.mean(total_power.get_data()[idx, :, :].reshape(
            -1, n_blocks, n_channels, n_timepts), axis=0))
        total_power_arr[i * n_blocks:(i + 1) * n_blocks, :, :] = block_avg

    # Rearrange axes of data to work for IEM pipeline
    total_power_arr = np.moveaxis(total_power_arr, 0, 1)
    return total_power_arr


def replicate_one_subj(subj):
    """Replicate one subject."""


def replicate_all_subjs(
    processed_dir=params.PROCESSED_DIR):
    """Replicate all subjects."""
    # Get all subject IDs
    subjs = sorted([f.split('_')[0] for f in os.listdir(
        processed_dir) if 'total_power' in f])

    # Process each subject's data
    for subj in subjs:
        replicate_one_subj(subj)
