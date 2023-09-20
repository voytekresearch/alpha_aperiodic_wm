import params
import numpy as np
import mne
import os


def diff_rust_vs_sparam_one_param(param, sparam_dir=params.SPARAM_DIR):
    """Compare spectral parameterization output from rust to that from FOOOF."""
    # Determine
    sparam_dirs = [sparam_dir, sparam_dir + '_fooof']

    # Get subject list
    subjects = [[f for f in os.listdir(
        d) if '.fif' not in f] for d in sparam_dirs]
    print(subjects)
    return


if __name__ == '__main__':
    # Determine difference between sparam output from rust and FOOOF
    diff_rust_vs_sparam_one_param('PW')