import params
import numpy as np
import mne
import os


def _diff_rust_vs_sparam_one_subj(
        subj, sparam_dirs, param, sparam_diff_dir=params.SPARAM_DIFF_DIR):
    """Compare spectral parameterization output from rust to that from FOOOF
    for one subject."""
    # See if data already exists
    os.makedirs(sparam_diff_dir, exist_ok=True)
    fname = f'{sparam_diff_dir}/{subj}_{param}_diff.npy'
    if os.path.exists(fname):
        return np.load(fname)

    # Extract data from fif files
    arrs = []
    for sparam_dir in sparam_dirs:
        # Load data
        fname = f'{sparam_dir}/{subj}_{param}_epo.fif'
        data = mne.read_epochs(fname, verbose=False).get_data()
        arrs.append(data)

    # Compute difference
    diff = np.subtract(arrs[0], arrs[1])

    # Determine which arrays are zero (0: neither, 1: arr1, 2: arr2, 3: both)
    zero_arr = 1 * (arrs[0] == 0) + 2 * (arrs[1] == 0)

    # Stack difference and zero arrays
    comp_arrs = np.stack((diff, zero_arr))

    # Save comparison arrays
    np.save(fname, comp_arrs)
    return comp_arrs


def diff_rust_vs_sparam_one_param(param, sparam_dir=params.SPARAM_DIR):
    """Compare spectral parameterization output from rust to that from FOOOF
    for one parameter."""
    # Determine sparam directories
    sparam_dirs = [sparam_dir, sparam_dir + '_fooof']

    # Get subject list
    subjects = [[f for f in os.listdir(
        d) if '.fif' not in f] for d in sparam_dirs]
    subjects = [s for s in subjects[0] if s in subjects[1]]

    # Loop through subjects
    for subject in subjects:
        _diff_rust_vs_sparam_one_subj(subject, sparam_dirs, param)
    return


if __name__ == '__main__':
    # Determine difference between sparam output from rust and FOOOF
    diff_rust_vs_sparam_one_param('PW')