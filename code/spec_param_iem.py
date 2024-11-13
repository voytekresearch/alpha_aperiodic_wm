"""Fit inverted encoding model (IEM) for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

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
    ctf_slopes, t_arrays = fit_model_desired_params(
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
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
        name="iem",
    )
