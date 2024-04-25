"""Fit inverted encoding model (IEM) for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import necessary modules
import matplotlib.pyplot as plt
from train_and_test_model import fit_model_desired_params
from plot_model_fits import (
    plot_model_fit_time_courses,
    compare_params_model_fit_time_courses,
    plot_model_fit_paired_ttest,
)

if __name__ == "__main__":
    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    param_names = {
        "total_power": "Alpha total power",
        "linOscAUC": "Alpha oscillatory power",
        "exponent": "Aperiodic exponent",
    }
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        model_fits_contrast=ctf_slopes_null,
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
    )

    # Plot paired t-tests of CTF slopes for the aperiodic exponent in first
    # 400 ms after presentation
    cmap = plt.get_cmap("Paired")
    plot_model_fit_paired_ttest(
        ctf_slopes["exponent"],
        t_arrays["exponent"],
        (0.0, 0.4),
        model_fits_shuffled=ctf_slopes_null["exponent"],
        palette=(cmap(3), cmap(2)),
        save_fname="exp_ctf_slope_paired_t-test.png",
    )

    # Plot paired t-tests of CTF slopes for alpha oscillatory power in WM
    pw_ctf_slope_fname = "pw_ctf_slope_paired_t-test.png"
    plot_model_fit_paired_ttest(
        ctf_slopes["linOscAUC"],
        t_arrays["linOscAUC"],
        "delay",
        model_fits_shuffled=ctf_slopes_null["linOscAUC"],
        palette=(cmap(1), cmap(0)),
        save_fname="pw_ctf_slope_paired_t-test.png",
    )

    # Plot CTF slope time courses for relevant comparisons of parameters from
    # spectral parameterization model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_model_desired_params(
        sp_params="all", verbose=False
    )
    compare_params_model_fit_time_courses(
        ctf_slopes, ctf_slopes_null, t_arrays
    )
