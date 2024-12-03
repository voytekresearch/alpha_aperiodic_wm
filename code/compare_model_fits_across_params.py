# Import necessary packages
import numpy as np
import os
import cmasher as cmr
from train_and_test_model import fit_model_desired_params
from fig3_compare_model_fits_across_tasks import plot_model_fit_time_courses
import params


def compare_params_model_fit_time_courses(
    model_fits_all_params,
    t_all_params,
    model_output_name="CTF slope",
    sparam_dir=params.SPARAM_DIR,
):
    """Compare model fit time courses for different parameters from spectral
    parameterization model."""
    # Get set of all parameters
    all_params = {
        f.split("_")[-2] for f in os.listdir(sparam_dir) if f.endswith(".fif")
    }

    # Sort and add total power to all parameters
    all_params = sorted(list(all_params) + ["total_power"])

    # Get sets of parameters to compare
    log_params = [p for p in all_params if "log" in p]
    lin_params = [p for p in all_params if "lin" in p]
    subj_params = [p for p in all_params if "Subj" in p]
    non_subj_params = [
        p
        for p in all_params
        if p in log_params + lin_params and p not in subj_params
    ]
    tot_params = [p for p in all_params if "tot" in p or "Tot" in p]
    osc_params = [p for p in all_params if "Osc" in p or "PW" in p]
    error_params = ["MSE", "R^2"]
    fit_params = [
        p for p in all_params if "AUC" not in p and p not in error_params
    ]
    fit_params.remove("total_power")
    comp_sets = [
        [log_params, lin_params],
        [subj_params, non_subj_params],
        [tot_params, osc_params],
        [error_params, fit_params],
    ]

    # Get palettes for comparison
    comp_palettes = ["PiYG", "BrBG", "PuOr", "RdBu"]

    # Define names and titles for comparisons
    comp_names = [
        "log_vs_lin",
        "subj_vs_non_subj",
        "tot_vs_osc",
        "error_vs_fit",
    ]
    comp_titles = [
        "Logarithmic vs linear difference",
        "Subject-specific alpha band or not",
        "Aperiodic-adjusted alpha power or not",
        "Spectral parameterization fit vs error",
    ]
    for comp_set, comp_palette, comp_name, comp_title in zip(
        comp_sets, comp_palettes, comp_names, comp_titles
    ):
        # Get palettes for comparison
        palettes = [
            cmr.take_cmap_colors(
                comp_palette, len(comp_set[0]), cmap_range=(0.0, 0.35)
            ),
            cmr.take_cmap_colors(
                comp_palette, len(comp_set[1]), cmap_range=(0.65, 1.0)
            ),
        ]

        # Plot model fit time courses for comparison
        plot_model_fit_time_courses(
            model_fits_all_params,
            t_all_params,
            model_output_name=model_output_name,
            param_sets=comp_set,
            palettes=palettes,
            name=comp_name,
            title=comp_title,
        )


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, t_arrays = fit_model_desired_params(verbose=False)
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        title="All parameters",
        palettes=["Set1"],
        plt_errorbars=True,
        name="iem",
    )
