# Import necessary modules
import numpy as np
import os
from itertools import combinations
from matplotlib import cm
import matplotlib.colors as mcolors
from train_and_test_model import (
    load_param_data,
    equalize_param_data_across_trial_blocks,
    get_subject_list,
)
from fig4_compare_model_fits_across_tasks import plot_model_fit_time_courses
import params


def prepare_param_data_for_all_subjs(
    param,
    param_dir,
    average=False,
    verbose=True,
    subjects_by_task=params.SUBJECTS_BY_TASK,
):
    """Load and structure parameterized data across subjects without fitting IEMs."""
    # Get list of subject IDs
    subjs = get_subject_list(param, param_dir)

    # Store outputs
    param_arrs_by_task = [[] for _ in range(len(subjects_by_task))]
    times_by_task = [[] for _ in range(len(subjects_by_task))]

    for subj in subjs:
        try:
            # Load parameterized data
            epochs, times, param_data = load_param_data(
                subj, param, param_dir, metadata=False
            )

            # Equalize and organize
            param_arr, _ = equalize_param_data_across_trial_blocks(
                epochs, times, param_data, average=average
            )

            if param_arr is None:
                if verbose:
                    print(f"Skipping {subj} due to NaNs in position bins")
                continue

            # Determine which task this subject belongs to
            experiment, subj_num = subj.split("_")
            task_idx = np.argmax(
                [
                    experiment == exp and int(subj_num) in ids
                    for exp, ids in params.SUBJECTS_BY_TASK
                ]
            )

            # Save structured data
            param_arrs_by_task[task_idx].append(param_arr)
            times_by_task[task_idx] = times

        except Exception as e:
            if verbose:
                print(f"Error processing {subj}: {e}")

    return param_arrs_by_task, times_by_task


def prepare_param_data_all_params(
    sp_params=tuple(params.PARAMS_TO_PLOT.keys()),
    sparam_dir=params.SPARAM_DIR,
    total_power_dir=params.TOTAL_POWER_DIR,
    average=False,
    verbose=True,
    subjects_by_task=params.SUBJECTS_BY_TASK,
):
    """
    Load and structure parameterized data for all parameters (including total power)
    across subjects without fitting IEMs.

    Returns
    -------
    param_data_all : dict
        Dictionary with keys for each parameter and values as lists of arrays per task.
    times_all : dict
        Dictionary with keys for each parameter and values as lists of time arrays per task.
    """
    # Get all parameters from the sparam_dir (everything except 'total_power')
    if sp_params is None or len(sp_params) == 0:
        sp_params = {
            f.split("_")[-2]
            for f in os.listdir(sparam_dir)
            if f.endswith(".fif")
        }
        sp_params.add("total_power")

    # Store outputs
    param_data_all = {}
    times_all = {}

    # Load param data
    for param in sorted(sp_params):
        # Choose correct directory
        param_dir = total_power_dir if param == "total_power" else sparam_dir

        if verbose:
            print(f"Preparing data for: {param}")

        param_arrs, t_arrs = prepare_param_data_for_all_subjs(
            param=param,
            param_dir=param_dir,
            average=average,
            verbose=verbose,
            subjects_by_task=subjects_by_task,
        )

        param_data_all[param] = param_arrs
        times_all[param] = t_arrs

    return param_data_all, times_all


def compute_param_correlations_across_time(
    param_data_all, times_all, params_to_plot=params.PARAMS_TO_PLOT, cmap=None
):
    """
    Compute correlation timecourses between every pair of parameters and
    prepare plotting metadata for each pair.

    Parameters
    ----------
    param_data_all : dict
        Keys are single-parameter names (e.g. "exponent", "linOscAUC", "total_power",
        "offset"). Values are lists (one per task) of arrays shaped (n_channels, n_blocks, n_timepts).
    times_all : dict
        Keys are single-parameter names; values are lists (one per task) of 1D time-vectors.
    params_to_plot : dict, default=params.PARAMS_TO_PLOT
        Mapping single-parameter → {"name": display string, "color": hex or RGB tuple}.
    cmap : str or None, default=None
        If a colormap name (e.g. 'viridis', 'plasma', 'tab10') is provided, each
        parameter-pair will be assigned its own color by sampling evenly from that colormap.
        If None, each pair’s color is the average of its two single-parameter colors.

    Returns
    -------
    model_fits_all_params : dict
        Keys are 2-tuples of parameter names. Values are lists (one per task)
        of 2D arrays shaped (n_subjects, n_timepts), each row a subject’s correlation timecourse.
    t_all_params : dict
        Same keys as model_fits_all_params; each value is the list (one per task) of time-vectors.
    corrs_to_plot : dict
        Keys are the same 2-tuples. Values are {"name": str, "color": RGB or RGBA} for plotting.
    """
    param_list = sorted(param_data_all.keys())
    param_pairs = list(combinations(param_list, 2))
    n_tasks = len(param_data_all[param_list[0]])

    # If the user passed a colormap name, retrieve the cmap object and sample it
    if cmap is not None:
        col = cm.get_cmap(cmap)
        n_pairs = len(param_pairs)
        sampled_colors = [col(i / float(n_pairs - 1)) for i in range(n_pairs)]
    else:
        sampled_colors = None

    model_fits = {pair: [] for pair in param_pairs}
    t_arrays = {pair: times_all[param_list[0]] for pair in param_pairs}

    for task_idx in range(n_tasks):
        n_subjs = len(param_data_all[param_list[0]][task_idx])

        # Reshape each parameter’s data into (n_samples × n_timepts) per subject
        reshaped = {}
        for param in param_list:
            reshaped[param] = []
            for subj_idx in range(n_subjs):
                arr = param_data_all[param][task_idx][subj_idx]
                flattened = arr.reshape(
                    -1, arr.shape[-1]
                )  # (n_samples, n_timepts)
                reshaped[param].append(flattened)

        for pi, pair in enumerate(param_pairs):
            X_list = reshaped[pair[0]]
            Y_list = reshaped[pair[1]]
            tc_per_subj = []
            for subj_idx in range(n_subjs):
                x = X_list[subj_idx]
                y = Y_list[subj_idx]
                n_timepts = x.shape[-1]
                timecourse = np.array(
                    [
                        np.corrcoef(x[:, t], y[:, t])[0, 1]
                        for t in range(n_timepts)
                    ]
                )
                tc_per_subj.append(timecourse)
            model_fits[pair].append(np.vstack(tc_per_subj))

    corrs_to_plot = {}
    for i, pair in enumerate(param_pairs):
        p1, p2 = pair
        name1, color1 = params_to_plot[p1]["name"], params_to_plot[p1]["color"]
        name2, color2 = params_to_plot[p2]["name"], params_to_plot[p2]["color"]

        if sampled_colors is not None:
            final_color = sampled_colors[i]
        else:
            rgb1 = mcolors.to_rgb(color1)
            rgb2 = mcolors.to_rgb(color2)
            final_color = tuple((np.array(rgb1) + np.array(rgb2)) / 2.0)

        corrs_to_plot[pair] = {
            "name": f"{name1} vs {name2}",
            "color": final_color,
        }

    return model_fits, t_arrays, corrs_to_plot


if __name__ == "__main__":
    # Add offset parameter to the list of parameters to plot
    params_to_plot = params.PARAMS_TO_PLOT.copy()
    params_to_plot["offset"] = {
        "name": "Aperiodic offset",
        "color": "#2cb37d",
    }

    # Load and prepare parameter data for all subjects
    param_data_all, times_all = prepare_param_data_all_params(
        sp_params=tuple(params_to_plot.keys())
    )

    # Compute pairwise correlations
    model_fits_all_params, t_all_params, corrs_to_plot = (
        compute_param_correlations_across_time(
            param_data_all,
            times_all,
            params_to_plot=params_to_plot,
            cmap="gist_earth",
        )
    )
    corrs_pairs = [
        ("exponent", "offset"),
        ("exponent", "linOscAUC"),
        ("linOscAUC", "total_power"),
    ]
    corrs_to_plot = {
        k: v
        for k, v in corrs_to_plot.items()
        if k in corrs_pairs or k[:-1] in corrs_pairs
    }

    # Plot the correlations
    plot_model_fit_time_courses(
        model_fits_all_params,
        t_all_params,
        params_to_plot=corrs_to_plot,
        name="correlations",
        model_output_name="Pearson r",
        save_fname="sfig2_correlate_params.pdf",
        plt_sig=False,
        zscore=False,
        plt_errorbars=True,
    )
