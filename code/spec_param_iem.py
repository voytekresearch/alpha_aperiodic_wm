"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import neccesary modules
import os
import matplotlib.pyplot as plt
import params
import time
from train_and_test_iem import plot_ctf_slope, plot_ctf_slope_paired_ttest, \
    train_and_test_all_subjs


def fit_iem_all_params(
        sparam_dir=params.SPARAM_DIR, total_power_dir=params.TOTAL_POWER_DIR,
        verbose=True):
    """Fit inverted encoding model (IEM) for total power and all parameters from
    spectral parameterization."""
    # Determine all parameters to fit IEM for
    sp_params = set([f.split('_')[-2] for f in os.listdir(
        sparam_dir) if f.endswith('.fif')])

    # Fit IEM for total power
    ctf_slopes_all_params, ctf_slopes_null_all_params = {}, {}
    _, tot_pw_ctf_slopes, tot_pw_ctf_slopes_null, _ = train_and_test_all_subjs(
        'total_power', total_power_dir)
    ctf_slopes_all_params['total_power'] = tot_pw_ctf_slopes
    ctf_slopes_null_all_params['total_power'] = tot_pw_ctf_slopes_null

    # Fit IEM for all parameters from spectral parameterization
    for sp_param in sp_params:
        # Start timer
        start = time.time()
        _, ctf_slopes, ctf_slopes_null, t_arrays = train_and_test_all_subjs(
            sp_param, sparam_dir)
        ctf_slopes_all_params[sp_param] = ctf_slopes
        ctf_slopes_null_all_params[sp_param] = ctf_slopes_null
        if verbose:
            print(f'Fit IEMs for {sp_param} in {time.time() - start:.2f} s')
    return ctf_slopes_all_params, ctf_slopes_null_all_params, t_arrays


def plot_ctf_slope_time_courses(
        ctf_slopes, ctf_slopes_null, t_arrays,
        subjects_by_task=params.SUBJECTS_BY_TASK, fig_dir=params.FIG_DIR,
        task_timings=params.TASK_TIMINGS):
    """Plot CTF slope time courses for total power and parameters from spectral
    parameterization."""
    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    for task_num, (experiment, _) in enumerate(subjects_by_task):
        ctf_slopes_fname = os.path.join(
            fig_dir, f'ctf_slopes_{experiment}_task{task_num}.png')
        ctf_slopes = {k: v[task_num] for k, v in ctf_slopes.items()}
        ctf_slopes_shuffled = {
            k: v[task_num] for k, v in ctf_slopes_null.items()}
        plot_ctf_slope(
            ctf_slopes, t_arrays[task_num], task_num,
            task_timings=task_timings[task_num],
            ctf_slopes_shuffled=ctf_slopes_shuffled, palette='Set1',
            save_fname=ctf_slopes_fname)


def plot_paired_ttests(ctf_slopes, ctf_slopes_null, t_arrays):
    """Plot paired t-tests of CTF slopes for desired parameters from spectral
    parameterization model."""
    # Plot paired t-tests of CTF slopes for the aperiodic exponent in first
    # 400 ms after presentation
    exp_ctf_slope_fname = f'{params.FIG_DIR}/exp_ctf_slope_paired_ttest.png'
    cmap = plt.get_cmap('Paired')
    plot_ctf_slope_paired_ttest(
        ctf_slopes['exponent'], t_arrays, (0.0, 0.4),
        ctf_slopes_shuffled=ctf_slopes_null['exponent'],
        palette=(cmap(3), cmap(2)), save_fname=exp_ctf_slope_fname)

    # Plot paired t-tests of CTF slopes for alpha oscillatory power in WM
    pw_ctf_slope_fname = f'{params.FIG_DIR}/pw_ctf_slope_paired_ttest.png'
    plot_ctf_slope_paired_ttest(
        ctf_slopes['PW'], t_arrays, 'WM',
        ctf_slopes_shuffled=ctf_slopes_null['PW'], palette=(cmap(1), cmap(0)),
        save_fname=pw_ctf_slope_fname)


if __name__ == '__main__':
    # Fit IEM for total power and all parameters from spectral parameterization
    # model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_iem_all_params()

    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    plot_ctf_slope_time_courses(ctf_slopes, ctf_slopes_null, t_arrays)

    # Plot paired t-tests of CTF slopes for parameters from spectral
    # parameterization model
    plot_paired_ttests(ctf_slopes, ctf_slopes_null, t_arrays)
