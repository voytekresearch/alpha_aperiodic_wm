# Import necessary modules
import os
import numpy as np
import pandas as pd
import params


def count_psds(sparam_dir=params.SPARAM_DIR, exclude_str=None, verbose=True):
    # Get folders in spectral parameterization directory
    sub_folders = [name for name in os.listdir(sparam_dir) if os.path.isdir(
        os.path.join(sparam_dir, name))]

    # Exclude folders if desired
    if exclude_str is not None:
        sub_folders = [name for name in sub_folders if exclude_str not in name]

    # For each folder, count number of PSDs
    num_psds = []
    for d in sub_folders:
        trial_csvs = sorted(os.listdir(os.path.join(sparam_dir, d)))
        df = pd.read_csv(os.path.join(sparam_dir, d, trial_csvs[0]))
        num_psds.append([len(trial_csvs),
                         len(set(df['channel'])),
                         len(set(df['timepoint']))])
    num_psds = np.array(num_psds)
    averages = np.mean(num_psds, axis=0)
    tot_psds = np.sum(np.prod(num_psds, axis=1))

    # If verbose, print counts
    if verbose:
        print_str = (f'Number of subjects: {len(sub_folders)}\n'
                     f'Average number of trials: {averages[0]:.1f}\n'
                     f'Average number of channels: {averages[1]:.1f}\n'
                     f'Average number of timepoints: {averages[2]:.1f}\n'
                     f'Total number of PSDs: {tot_psds}')
        print(print_str)


if __name__ == '__main__':
    count_psds(exclude_str='SB')
