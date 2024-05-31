"""Fit circular ridge regression for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from EEG data."""

# Import necessary modules
import numpy as np
from train_and_test_model import fit_model_desired_params
from plot_model_fits import plot_model_fit_time_courses
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
        single_trials=True,
        output_dir=params.SINGLE_TRIAL_CRR_DIR,
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
        name="single_trial_crr",
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
    )
