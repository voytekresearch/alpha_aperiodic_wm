"""Decode spatial location from alpha oscillatory power and aperiodic exponent
using same inverted encoding model (IEM) and EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import necessary modules
import os
import time
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import params
from train_and_test_iem import (
    plot_ctf_slope,
    plot_ctf_slope_paired_ttest,
    train_and_test_all_subjs,
)


def fit_iem_all_params(
    sparam_dir=params.SPARAM_DIR,
    total_power_dir=params.TOTAL_POWER_DIR,
    trial_split_criterion=None,
    verbose=True,
    output_dir=params.IEM_OUTPUT_DIR,
):
    """Fit inverted encoding model (IEM) for total power and all parameters from
    spectral parameterization."""
    # Determine all parameters to fit IEM for
    sp_params = {
        f.split("_")[-2] for f in os.listdir(sparam_dir) if f.endswith(".fif")
    }

    # Fit IEM for total power
    ctf_slopes_all_params, ctf_slopes_null_all_params = {}, {}
    if total_power_dir is not None:
        (
            tot_pw_ctf_slopes,
            tot_pw_ctf_slopes_null,
            _,
        ) = train_and_test_all_subjs(
            "total_power", total_power_dir, output_dir=output_dir
        )
        ctf_slopes_all_params["total_power"] = tot_pw_ctf_slopes
        ctf_slopes_null_all_params["total_power"] = tot_pw_ctf_slopes_null

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
        if verbose:
            print(f"Fit IEMs for {sp_param} in {time.time() - start:.2f} s")
    return ctf_slopes_all_params, ctf_slopes_null_all_params, t


def plot_ctf_slope_time_courses(
    ctf_slopes_all_params,
    ctf_slopes_null_all_params,
    t,
    param_sets=None,
    palettes=None,
    name="",
    title="",
    subjects_by_task=params.SUBJECTS_BY_TASK,
    fig_dir=params.FIG_DIR,
    task_timings=params.TASK_TIMINGS,
):
    """Plot CTF slope time courses for total power and parameters from spectral
    parameterization."""
    # Set default parameter sets and palettes
    if param_sets is None:
        param_sets = [list(ctf_slopes_all_params.keys())]
    if palettes is None:
        palettes = ["rocket"]

    # Create a GridSpec with one row and the number of tasks as columns
    num_tasks = len(subjects_by_task)
    fig = plt.figure(figsize=(48, 20), constrained_layout=True)
    gs = gridspec.GridSpec(2, num_tasks // 2 + 1, figure=fig)

    # Plot CTF slope time courses for parameters from spectral parameterization
    # model
    for task_num in range(len(subjects_by_task)):
        ctf_slopes_one_task = {
            k: v[task_num] for k, v in ctf_slopes_all_params.items()
        }
        ctf_slopes_shuffled = {
            k: v[task_num] for k, v in ctf_slopes_null_all_params.items()
        }

        # Get the corresponding subplot from the GridSpec
        ax = fig.add_subplot(gs[task_num % 2, task_num // 2])

        # Plot CTF slope time courses for each parameter set and palette
        for i, (param_set, palette) in enumerate(zip(param_sets, palettes)):
            # Get CTF slopes for parameter set
            ctf_slopes_one_param_set = {
                k: v for k, v in ctf_slopes_one_task.items() if k in param_set
            }
            ctf_slopes_shuffled_one_param_set = {
                k: v for k, v in ctf_slopes_shuffled.items() if k in param_set
            }

            # Plot CTF slope time courses for parameter set and palette
            plt_timings = i == len(param_sets) - 1
            plot_ctf_slope(
                ctf_slopes_one_param_set,
                t[task_num],
                task_num,
                task_timings=task_timings[task_num],
                ctf_slopes_shuffled=ctf_slopes_shuffled_one_param_set,
                palette=palette,
                plot_timings=plt_timings,
                plot_errorbars=False,
                ax=ax,
            )

    # Get legend handles and labels from last axis
    handles, labels = ax.get_legend_handles_labels()
    dup_idx = np.where(pd.DataFrame(labels).duplicated(keep=False))[0]
    remove_idx = dup_idx[1 : 1 + len(dup_idx) // 2]
    handles = [h for i, h in enumerate(handles) if i not in remove_idx]
    labels = [l for i, l in enumerate(labels) if i not in remove_idx]
    ax_legend = fig.add_subplot(gs[-1, -1])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center", fontsize=24)

    # Add title
    fig.suptitle(title, fontsize=64, fontweight="bold", y=1.05)

    # Save figure
    os.makedirs(fig_dir, exist_ok=True)
    ctf_slopes_fname = f"{fig_dir}/ctf_slopes.png"
    if len(name) > 0:
        ctf_slopes_fname = ctf_slopes_fname.replace(".", f"_{name}.")
    plt.savefig(ctf_slopes_fname, dpi=300, bbox_inches="tight")
    plt.close()


def compare_params_ctf_time_courses(
    ctf_slopes_all_params,
    ctf_slopes_null_all_params,
    t,
):
    """Compare CTF slope time courses for different parameters from spectral
    parameterization model."""
    all_params = list(
        {
            f.split("_")[-2]
            for f in os.listdir(params.SPARAM_DIR)
            if f.endswith(".fif")
        }
    )
    all_params = sorted(all_params + ["total_power"])
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
    comp_palettes = ["PiYG", "BrBG", "PuOr", "RdBu"]
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
        plot_ctf_slope_time_courses(
            ctf_slopes_all_params,
            ctf_slopes_null_all_params,
            t,
            param_sets=comp_set,
            palettes=[
                cmr.take_cmap_colors(
                    comp_palette, len(comp_set[0]), cmap_range=(0.0, 0.35)
                ),
                cmr.take_cmap_colors(
                    comp_palette, len(comp_set[1]), cmap_range=(0.65, 1.0)
                ),
            ],
            name=comp_name,
            title=comp_title,
        )


def plot_paired_ttests(ctf_slopes_all_params, ctf_slopes_null_all_params, t):
    """Plot paired t-tests of CTF slopes for desired parameters from spectral
    parameterization model."""
    # Plot paired t-tests of CTF slopes for the aperiodic exponent in first
    # 400 ms after presentation
    exp_ctf_slope_fname = f"{params.FIG_DIR}/exp_ctf_slope_paired_t-test.png"
    cmap = plt.get_cmap("Paired")
    plot_ctf_slope_paired_ttest(
        ctf_slopes_all_params["exponent"],
        t,
        (0.0, 0.4),
        ctf_slopes_shuffled=ctf_slopes_null_all_params["exponent"],
        palette=(cmap(3), cmap(2)),
        save_fname=exp_ctf_slope_fname,
    )

    # Plot paired t-tests of CTF slopes for alpha oscillatory power in WM
    pw_ctf_slope_fname = f"{params.FIG_DIR}/pw_ctf_slope_paired_t-test.png"
    plot_ctf_slope_paired_ttest(
        ctf_slopes_all_params["PW"],
        t,
        "delay",
        ctf_slopes_shuffled=ctf_slopes_null_all_params["PW"],
        palette=(cmap(1), cmap(0)),
        save_fname=pw_ctf_slope_fname,
    )


if __name__ == "__main__":
    # Fit IEM for total power and all parameters from spectral parameterization
    # model
    ctf_slopes, ctf_slopes_null, t_arrays = fit_iem_all_params(verbose=False)

    # Plot CTF slope time courses for all parameters from spectral
    # parameterization model
    plot_ctf_slope_time_courses(
        ctf_slopes, ctf_slopes_null, t_arrays, title="All parameters"
    )

    # Plot CTF slope time courses for relevant comparisons of parameters from
    # spectral parameterization model
    compare_params_ctf_time_courses(ctf_slopes, ctf_slopes_null, t_arrays)

    # Plot paired t-tests of CTF slopes for parameters from spectral
    # parameterization model
    plot_paired_ttests(ctf_slopes, ctf_slopes_null, t_arrays)
