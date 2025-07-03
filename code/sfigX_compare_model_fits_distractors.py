# Import necessary modules
import numpy as np
from train_and_test_model import fit_model_desired_params
from fig4_compare_model_fits_across_tasks import plot_model_fit_time_courses
import params

if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model on distractors
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        distractors=True,
        verbose=False,
    )
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        title="All parameters",
        plt_errorbars=True,
        name="iem",
        save_fname="sfigX_iem_spec_param_distractors.pdf",
    )
