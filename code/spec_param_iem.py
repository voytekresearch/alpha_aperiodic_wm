"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import neccesary modules
import os
import params
from train_and_test_iem import plot_ctf_slope, train_and_test_all_subjs

if __name__ == '__main__':
    # Reproduce total alpha power IEM
    _, tot_pw_ctf_slopes, _ = train_and_test_all_subjs(
        'total_power', params.TOTAL_POWER_DIR)

    # Decode spatial location from alpha oscillatory power
    _, pw_ctf_slopes, t_arrays = train_and_test_all_subjs(
        'PW', params.SPARAM_DIR)

    # Decode spatial location from aperiodic exponent
    _, exp_ctf_slopes, _ = train_and_test_all_subjs(
        'exponent', params.SPARAM_DIR)

    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    for task_num, (experiment, _) in enumerate(params.SUBJECTS_BY_TASK):
        ctf_slopes_fname = os.path.join(
            params.FIG_DIR, f'ctf_slopes_{experiment}_task{task_num}.png')
        ctf_slopes = {
            'Alpha total power': -tot_pw_ctf_slopes[task_num],
            'Alpha oscillatory power': -pw_ctf_slopes[task_num],
            'Aperiodic exponent': -exp_ctf_slopes[task_num]}
        plot_ctf_slope(
            ctf_slopes, t_arrays[task_num], task_num,
            task_timings=params.TASK_TIMINGS[task_num], palette='Set1',
            save_fname=ctf_slopes_fname)
