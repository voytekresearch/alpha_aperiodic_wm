"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/"""

# Import neccesary modules
import os
import params
from train_and_test_iem import plot_ctf_slope, train_and_test_all_subjs

if __name__ == '__main__':
    # Reproduce total alpha power IEM
    tot_pw_channel_offsets, tot_pw_ctf_slopes, t_arr = train_and_test_all_subjs(
        'total_power')

    # Decode spatial location from alpha oscillatory power
    pw_channel_offsets, pw_ctf_slopes, _ = train_and_test_all_subjs('PW')

    # Decode spatial location from aperiodic exponent
    exp_channel_offsets, exp_ctf_slopes, _ = train_and_test_all_subjs(
        'exponent')

    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    ctf_slopes_fname = os.path.join(params.FIG_DIR, 'ctf_slopes')
    ctf_slopes = {
        'Alpha total power': tot_pw_ctf_slopes,
        'Alpha oscillatory power': pw_ctf_slopes,
        'Aperiodic exponent': exp_ctf_slopes}
    plot_ctf_slope(ctf_slopes, t_arr, save_fname=ctf_slopes_fname)
