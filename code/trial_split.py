"""Split trials based on selection criteria and re-train IEMs on those trial 
splits."""

# Import necessary modules
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
            f"Bottom {split_dct['bottom_frac']:.0%}",
            f"Top {split_dct['top_frac']:.0%}",
        ],
        title=name,
        name=name.lower().replace(" ", "_"),
    )


if __name__ == "__main__":
    # Split trials based on exponent change
    print("\nEXPONENT CHANGE")
    exponent_change_dct = {
        "param": "exponent",
        "t_window": "stimulus",
        "baseline_t_window": "baseline",
        "bottom_frac": 0.25,
        "top_frac": 0.25,
        "channels": ("O1", "O2"),
    }
    split_trials_on_param(
        ["exponent", "linOscAUC", "total_power"],
        exponent_change_dct,
        "Exponent Change",
    )

    # Split trials based on alpha change
    print("\nALPHA CHANGE")
    alpha_change_dct = {
        "param": "linOscAUC",
        "t_window": "delay",
        "baseline_t_window": "baseline",
        "bottom_frac": 0.25,
        "top_frac": 0.25,
        "channels": ("O1", "O2"),
    }
    split_trials_on_param(
        ["exponent", "linOscAUC", "total_power"],
        alpha_change_dct,
        "Alpha Change",
    )

    # Split trials on total power change
    print("\nTOTAL POWER CHANGE")
    total_power_change_dct = {
        "param": "total_power",
        "param_dir": params.TOTAL_POWER_DIR,
        "t_window": "delay",
        "baseline_t_window": "baseline",
        "bottom_frac": 0.25,
        "top_frac": 0.25,
        "channels": ("O1", "O2"),
    }
    split_trials_on_param(
        ["exponent", "linOscAUC", "total_power"],
        total_power_change_dct,
        "Total Power Change",
    )

    # Split trials based on N1 amplitude
    print("\nHIGH/LOW N1 AMPLITUDE")
    error_dct = {
        "param": "n1amp",
        "param_dir": params.ERP_DIR,
        "t_window": None,
        "bottom_frac": 0.25,
        "top_frac": 0.25,
    }
    split_trials_on_param(
        ["exponent", "linOscAUC", "total_power"],
        error_dct,
        "N1 Amplitude",
    )

    # Split trials based on behavior
    print("\nHIGH/LOW ERROR")
    error_dct = {
        "param": "error",
        "t_window": None,
        "bottom_frac": 0.25,
        "top_frac": 0.25,
        "metadata": True,
    }
    split_trials_on_param(
        ["exponent", "linOscAUC", "total_power"],
        error_dct,
        "Performance",
    )
