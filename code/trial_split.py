"""Split trials based on selection criteria and re-train IEMs on those trial 
splits."""
# Import necessary modules
import os
import numpy as np
from spec_param_iem import (
    fit_iem_desired_params,
    plot_ctf_slope_time_courses,
)
import params

if __name__ == "__main__":
    # Get all parameters from spectral parameterization
    sp_params = {
        f.split("_")[-2]
        for f in os.listdir(params.SPARAM_DIR)
        if f.endswith(".fif")
    }
    lin_osc_auc_params = [p for p in sp_params if "linOscAUC" in p]
    # Split trials based on exponent change
    exponent_change_dct = {
        "param": "exponent",
        "t_window": "delay",
        "baseline_t_window": "baseline",
    }
    (
        ctf_slopes_exp_change,
        ctf_slopes_null_exp_change,
        t,
    ) = fit_iem_desired_params(
        sp_params=lin_osc_auc_params,
        total_power_dir=None,
        output_dir=params.TRIAL_SPLIT_DIR,
        trial_split_criterion=exponent_change_dct,
    )

    # Plot CTF slopes for exponent change
    high_exp_change = {
        k: [np.array([s["high"] for s in task]) for task in v]
        for k, v in ctf_slopes_exp_change.items()
    }
    low_exp_change = {
        k: [np.array([s["low"] for s in task]) for task in v]
        for k, v in ctf_slopes_exp_change.items()
    }
    plot_ctf_slope_time_courses(
        high_exp_change,
        t,
        ctf_slopes_contrast=low_exp_change,
        contrast_label="Exponent Change?",
        contrast_vals=["Bottom 50%", "Top 50%"],
        title="Exponent Change",
        name="exp_change",
    )
