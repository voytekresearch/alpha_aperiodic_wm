"""Fit circular ridge regression for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from EEG data."""

# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from train_and_test_model import fit_model_desired_params
from plot_model_fits import (
    plot_model_fit_time_courses,
    compare_params_model_fit_time_courses,
    plot_model_fit_paired_ttest,
)
import params

if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot circular correlation coefficient time courses for desired parameters
    # from spectral parameterization model
    circ_corrcoefs, circ_corrcoefs_null, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        method="crr",
        verbose=False,
        output_dir=params.CRR_OUTPUT_DIR,
    )
    param_names = {
        "total_power": "Alpha total power",
        "linOscAUC": "Alpha oscillatory power",
        "exponent": "Aperiodic exponent",
    }
    plot_model_fit_time_courses(
        circ_corrcoefs,
        t_arrays,
        model_fits_contrast=circ_corrcoefs_null,
        model_output_name="Circular correlation coefficient",
        name="crr",
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
    )

    # Plot paired t-tests of circular correlation coefficients for the aperiodic
    # exponent in first 400 ms after presentation
    cmap = plt.get_cmap("Paired")
    plot_model_fit_paired_ttest(
        circ_corrcoefs["exponent"],
        t_arrays["exponent"],
        (0.0, 0.4),
        model_fits_shuffled=circ_corrcoefs_null["exponent"],
        model_output_name="Circular correlation coefficient",
        palette=(cmap(3), cmap(2)),
        save_fname="exp_rc_paired_t-test.png",
    )

    # Plot paired t-tests of circular correlation coefficients for alpha
    # oscillatory power in WM
    pw_circ_corrcoef_fname = "pw_rc_paired_t-test.png"
    plot_model_fit_paired_ttest(
        circ_corrcoefs["linOscAUC"],
        t_arrays["linOscAUC"],
        "delay",
        model_fits_shuffled=circ_corrcoefs_null["linOscAUC"],
        model_output_name="Circular correlation coefficient",
        palette=(cmap(1), cmap(0)),
        save_fname="pw_rc_paired_t-test.png",
    )

    # Plot circular correlation coefficient time courses for relevant
    # comparisons of parameters from spectral parameterization model
    circ_corrcoefs, circ_corrcoefs_null, t_arrays = fit_model_desired_params(
        sp_params="all", verbose=False
    )
    compare_params_model_fit_time_courses(
        circ_corrcoefs, circ_corrcoefs_null, t_arrays
    )
