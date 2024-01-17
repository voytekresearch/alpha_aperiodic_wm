"""Split trials based on selection criteria and re-train IEMs on those trial 
splits."""
# Import necessary modules
import os
import numpy as np
from train_and_test_iem import fit_iem_desired_params
from plot_iem_results import plot_ctf_slope_time_courses
import params


def split_trials_on_param(
    iem_params, split_dct, name, trial_split_dir=params.TRIAL_SPLIT_DIR
):
    """Test whether trial splitting affects CTF slopes from training IEMs on
    desired parameters."""
    # Split trials based on exponent change
    (
        ctf_slopes,
        _,
        t,
        # Fit IEMs with trial split on exponent change
    ) = fit_iem_desired_params(
        sp_params=iem_params,
        total_power_dir=None,
        output_dir=trial_split_dir,
        trial_split_criterion=split_dct,
    )

    # Plot CTF slopes for split
    split_high = {
        k: [np.array([s["high"] for s in task]) for task in v]
        for k, v in ctf_slopes.items()
    }
    split_low = {
        k: [np.array([s["low"] for s in task]) for task in v]
        for k, v in ctf_slopes.items()
    }
    plot_ctf_slope_time_courses(
        split_high,
        t,
        ctf_slopes_contrast=split_low,
        contrast_label=f"{name}?",
        contrast_vals=[
            f"Bottom {split_dct['bottom_frac']:%}%",
            f"Top {split_dct['top_frac']:%}%",
        ],
        title=name,
        name=name.lower().replace(" ", "_"),
    )


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
        "t_window": "stimulus",
        "baseline_t_window": "baseline",
        "bottom_frac": 0.25,
        "top_frac": 0.25,
        "channels": ("O1", "O2"),
    }
    split_trials_on_param(
        lin_osc_auc_params, exponent_change_dct, "Exponent Change"
    )
