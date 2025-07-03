# Import necessary modules
import numpy as np
import params
from train_and_test_model import fit_model_desired_params
from fig4_compare_model_fits_across_tasks import plot_model_fit_time_courses

if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["offset", "exponent"],
        verbose=False,
    )
    params_to_plot = {
        "offset": {
            "dir": params.SPARAM_DIR,
            "name": "Aperiodic offset",
            "color": "#2cb37d",
        }
    }
    params_to_plot["exponent"] = params.PARAMS_TO_PLOT["exponent"].copy()
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        plt_errorbars=True,
        params_to_plot=params_to_plot,
        name="iem",
        save_fname="sfig1_plot_model_fits_offset.pdf",
    )
