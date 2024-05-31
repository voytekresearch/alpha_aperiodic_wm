"""Fit inverted encoding model (IEM) for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from single-trial EEG data."""

# Import necessary modules
import numpy as np
from train_and_test_model import fit_model_desired_params
from plot_model_fits import plot_model_fit_time_courses
import params

if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
        single_trials=True,
        output_dir=params.SINGLE_TRIAL_IEM_DIR,
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
        name="single_trial_iem",
    )
