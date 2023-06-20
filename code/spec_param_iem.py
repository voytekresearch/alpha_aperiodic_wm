"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import neccesary modules
import os
import params
from train_and_test_iem import plot_ctf_slope, train_and_test_all_subjs

if __name__ == '__main__':
    # Reproduce total alpha power IEM
    _, tot_pw_ctf_slopes, _ = train_and_test_all_subjs('total_power')

    # Decode spatial location from alpha oscillatory power
    _, pw_ctf_slopes, t_arr = train_and_test_all_subjs(
        'PW', processed_dir=params.SPARAM_DIR)

    # Decode spatial location from aperiodic exponent
    _, exp_ctf_slopes, _ = train_and_test_all_subjs(
        'exponent', processed_dir=params.SPARAM_DIR)

    # Extract experiments from slopes
    experiments = list(tot_pw_ctf_slopes.keys())
    assert experiments == list(pw_ctf_slopes.keys())
    assert experiments == list(exp_ctf_slopes.keys())

    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    for experiment in experiments:
        ctf_slopes_fname = os.path.join(
            params.FIG_DIR, f'ctf_slopes_{experiment}.png')
        ctf_slopes = {
            'Alpha total power': -tot_pw_ctf_slopes[experiment],
            'Alpha oscillatory power': -pw_ctf_slopes[experiment],
            'Aperiodic exponent': -exp_ctf_slopes[experiment]}
        plot_ctf_slope(
            ctf_slopes, t_arr, palette='Set1', save_fname=ctf_slopes_fname)
