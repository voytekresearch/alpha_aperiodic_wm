"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import necessary modules
import os
import time
import params
from train_and_test_iem import train_and_test_all_subjs
from plot_iem_results import (
    plot_ctf_slope_time_courses,
    compare_params_ctf_time_courses,
    plot_paired_ttests,
)


def fit_iem_desired_params(
    sp_params="all",
    sparam_dir=params.SPARAM_DIR,
    total_power_dir=params.TOTAL_POWER_DIR,
    trial_split_criterion=None,
    verbose=True,
    output_dir=params.IEM_OUTPUT_DIR,
):
    """Fit inverted encoding model (IEM) for total power and all parameters from
    spectral parameterization."""
    # Determine all parameters to fit IEM for
    if sp_params == "all":
        sp_params = {
            f.split("_")[-2]
            for f in os.listdir(sparam_dir)
            if f.endswith(".fif")
        }

    # Fit IEM for total power
    ctf_slopes_all_params = {}
    ctf_slopes_null_all_params = {}
    t_all_params = {}
    if total_power_dir is not None:
        (
            tot_pw_ctf_slopes,
            tot_pw_ctf_slopes_null,
            tot_pw_t,
        ) = train_and_test_all_subjs(
            "total_power", total_power_dir, output_dir=output_dir
        )
        ctf_slopes_all_params["total_power"] = tot_pw_ctf_slopes
        ctf_slopes_null_all_params["total_power"] = tot_pw_ctf_slopes_null
        t_all_params["total_power"] = tot_pw_t

    # Fit IEM for all parameters from spectral parameterization
    for sp_param in sp_params:
        # Start timer
        start = time.time()
        (
            ctf_slopes_one_param,
            ctf_slopes_null_one_param,
            t,
        ) = train_and_test_all_subjs(
            sp_param,
            sparam_dir,
            trial_split_criterion=trial_split_criterion,
            output_dir=output_dir,
        )
        ctf_slopes_all_params[sp_param] = ctf_slopes_one_param
        ctf_slopes_null_all_params[sp_param] = ctf_slopes_null_one_param
        t_all_params[sp_param] = t
        if verbose:
            print(f"Fit IEMs for {sp_param} in {time.time() - start:.2f} s")
    return ctf_slopes_all_params, ctf_slopes_null_all_params, t_all_params


if __name__ == "__main__":
    # Fit IEM for total power and all parameters from spectral parameterization
    # model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_iem_desired_params(
        verbose=False
    )

    # Plot CTF slope time courses for all parameters from spectral
    # parameterization model
    plot_ctf_slope_time_courses(
        ctf_slopes,
        t_arrays,
        ctf_slopes_contrast=ctf_slopes_null,
        title="All parameters",
    )

    # Plot CTF slope time courses for relevant comparisons of parameters from
    # spectral parameterization model
    compare_params_ctf_time_courses(ctf_slopes, ctf_slopes_null, t_arrays)

    # Plot paired t-tests of CTF slopes for parameters from spectral
    # parameterization model
    plot_paired_ttests(ctf_slopes, ctf_slopes_null, t_arrays)
