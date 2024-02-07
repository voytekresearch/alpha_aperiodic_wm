"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import necessary modules
import matplotlib.pyplot as plt
from train_and_test_iem import fit_iem_desired_params
from plot_iem_results import (
    plot_ctf_slope_time_courses,
    compare_params_ctf_time_courses,
    plot_paired_ttests,
)

if __name__ == "__main__":
    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_iem_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"], verbose=False
    )
    param_names = {
        "total_power": "Alpha total power",
        "linOscAUC": "Alpha oscillatory power",
        "exponent": "Aperiodic exponent",
    }
    plot_ctf_slope_time_courses(
        ctf_slopes,
        t_arrays,
        ctf_slopes_contrast=ctf_slopes_null,
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
    )

    # Plot paired t-tests of CTF slopes for parameters from spectral
    # parameterization model
    plot_paired_ttests(ctf_slopes, ctf_slopes_null, t_arrays)

    # Plot CTF slope time courses for relevant comparisons of parameters from
    # spectral parameterization model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_iem_desired_params(
        sp_params="all", verbose=False
    )
    compare_params_ctf_time_courses(ctf_slopes, ctf_slopes_null, t_arrays)
