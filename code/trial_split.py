"""Split trials based on selection criteria and re-train IEMs on those trial 
splits."""
# Import necessary modules
from spec_param_iem import fit_iem_all_params, compare_params_ctf_time_courses
import params

if __name__ == "__main__":
    # Split trials based on exponent change
    exponent_change_dct = {
        "param": "exponent",
        "t_window": "delay",
        "baseline_t_window": "baseline",
    }
    ctf_slopes_exp_change, ctf_slopes_null_exp_change, t = fit_iem_all_params(
        total_power_dir=None,
        output_dir=params.TRIAL_SPLIT_DIR,
        trial_split_criterion=exponent_change_dct,
    )

    # Plot CTF slopes for exponent change
    compare_params_ctf_time_courses(
        ctf_slopes_exp_change, ctf_slopes_null_exp_change, t
    )
