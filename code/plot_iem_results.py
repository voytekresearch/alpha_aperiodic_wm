"""Functions to plot results from IEM analyses."""

# Import necessary modules
import cmasher as cmr
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import params


def plot_ctf_slope(
    ctf_slopes,
    t_arrays,
    task_num,
    task_timings,
    ctf_slopes_contrast=None,
    param_names=None,
    contrast_label="Shuffled location labels?",
    contrast_vals=("No", "Yes"),
    palette=None,
    save_fname=None,
    plot_timings=True,
    plot_errorbars=False,
    ax=None,
):
    """Plot channel tuning function (CTF) across time for multiple
    parameters.

    Parameters
    ----------
    ctf_slopes : dict
        Dictionary containing CTF slopes for each parameter. Keys are parameter
        names and values are arrays containing CTF slopes.
    t_arr : np.ndarray
        Array containing time points.
    palette : dict (default: None)
        Dictionary containing colors for each parameter.  Keys are parameter
        names and values are colors.  If None, default colors will be used.
    save_fname : str (default: None)
        File name to save figure to.  If None, figure will not be saved.
    """
    # Set default parameter names
    if param_names is None:
        param_names = {k: k for k in ctf_slopes.keys()}

    # Make empty list for CTF slope DataFrames
    ctf_slopes_dfs = []

    # Make DataFrame of CTF slopes by time for each parameter
    for t_one_param, (param, ctf_slopes_one_param) in zip(
        t_arrays.values(), ctf_slopes.items()
    ):
        n = ctf_slopes_one_param.shape[0]
        one_param_df = pd.DataFrame(ctf_slopes_one_param, columns=t_one_param)
        one_param_df["Parameter"] = param_names[param]
        one_param_df = one_param_df.melt(
            id_vars=["Parameter"], var_name="Time (s)", value_name="CTF slope"
        )
        one_param_df[contrast_label] = contrast_vals[0]

        # Add DataFrame of CTF slopes for shuffled location labels if desired
        if ctf_slopes_contrast is not None:
            one_param_df_contrast = pd.DataFrame(
                ctf_slopes_contrast[param], columns=t_one_param
            )
            one_param_df_contrast["Parameter"] = param_names[param]
            one_param_df_contrast = one_param_df_contrast.melt(
                id_vars=["Parameter"],
                var_name="Time (s)",
                value_name="CTF slope",
            )
            one_param_df_contrast[contrast_label] = contrast_vals[1]
            one_param_df = pd.concat(
                (one_param_df, one_param_df_contrast), axis=0
            )
        ctf_slopes_dfs.append(one_param_df)

    # Combine DataFrames of CTF slopes for each parameter into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()

    # Plot CTF slope time course for each parameter
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    ctf_slopes_big_df["CTF slope"] = -ctf_slopes_big_df["CTF slope"]
    ci = 95 if plot_errorbars else None
    ax = sns.lineplot(
        data=ctf_slopes_big_df,
        hue="Parameter",
        x="Time (s)",
        y="CTF slope",
        style=contrast_label,
        palette=palette,
        legend="brief",
        ci=ci,
        ax=ax,
    )

    # Plot aesthetics
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
    if task_num != 0 or not save_fname:
        legend.remove()
    _, _, _, ymax = ax.axis()
    if plot_timings:
        ax.axvline(0.0, c="gray", ls="--")
        ax.text(0.03, ymax, "Stimulus onset", va="bottom", ha="right", size=24)
        ax.axvline(task_timings[0], c="gray", ls="--")
        offset_x, offset_ha = task_timings[0] + 0.03, "left"
        if task_timings[0] > 0.75:
            offset_x, offset_ha = task_timings[0], "center"
        ax.text(
            offset_x,
            ymax,
            "Stimulus offset",
            va="bottom",
            ha=offset_ha,
            size=24,
        )
        ax.axvline(task_timings[1], c="gray", ls="--")
        ax.text(
            task_timings[1],
            ymax,
            "Free response",
            va="bottom",
            ha="center",
            size=24,
        )
    ax.set_title(
        f"Task {task_num + 1} (n = {n})",
        fontsize=48,
        fontweight="bold",
        y=1.08,
    )
    ax.set_xlabel("Time (s)", size=28)
    ax.set_ylabel("CTF slope", size=28)
    ax.tick_params(labelsize=20)
    sns.despine(ax=ax)

    # Save if desired
    if save_fname:
        plt.savefig(save_fname, bbox_inches="tight", dpi=300)
        plt.close()


def plot_ctf_slope_time_courses(
    ctf_slopes_all_params,
    t_all_params,
    param_sets=None,
    palettes=None,
    param_names=None,
    name="",
    title="",
    ctf_slopes_contrast=None,
    contrast_label=None,
    contrast_vals=None,
    plt_errorbars=False,
    subjects_by_task=params.SUBJECTS_BY_TASK,
    fig_dir=params.FIG_DIR,
    task_timings=params.TASK_TIMINGS,
):
    """Plot CTF slope time courses for total power and parameters from spectral
    parameterization."""
    # Set default parameter sets, palettes, and parameter names
    if param_sets is None:
        param_sets = [list(ctf_slopes_all_params.keys())]
    if palettes is None:
        palettes = ["rocket"]
    if param_names is None:
        param_names = {k: k for k in ctf_slopes_all_params.keys()}

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
        if ctf_slopes_contrast is not None:
            ctf_slopes_contrast_one_task = {
                k: v[task_num] for k, v in ctf_slopes_contrast.items()
            }
        t = {k: v[task_num] for k, v in t_all_params.items()}

        # Get the corresponding subplot from the GridSpec
        ax = fig.add_subplot(gs[task_num % 2, task_num // 2])

        # Plot CTF slope time courses for each parameter set and palette
        for i, (param_set, palette) in enumerate(zip(param_sets, palettes)):
            # Get CTF slopes for parameter set
            ctf_slopes_one_param_set = {
                k: v for k, v in ctf_slopes_one_task.items() if k in param_set
            }
            if ctf_slopes_contrast is not None:
                ctf_slopes_contrast_one_param_set = {
                    k: v
                    for k, v in ctf_slopes_contrast_one_task.items()
                    if k in param_set
                }
            t_one_param_set = {k: v for k, v in t.items() if k in param_set}

            # Plot CTF slope time courses for parameter set and palette
            plt_timings = i == len(param_sets) - 1
            param_names_param_set = {k: param_names[k] for k in param_set}
            kwargs = {
                "palette": palette,
                "param_names": param_names_param_set,
                "plot_timings": plt_timings,
                "plot_errorbars": plt_errorbars,
                "ax": ax,
            }
            if ctf_slopes_contrast is not None:
                kwargs["ctf_slopes_contrast"] = (
                    ctf_slopes_contrast_one_param_set
                )
            if contrast_label is not None:
                kwargs["contrast_label"] = contrast_label
            if contrast_vals is not None:
                kwargs["contrast_vals"] = contrast_vals
            plot_ctf_slope(
                ctf_slopes_one_param_set,
                t_one_param_set,
                task_num,
                task_timings[task_num],
                **kwargs,
            )

    # Get legend handles and labels from last axis
    handles, labels = ax.get_legend_handles_labels()
    dup_idx = np.where(pd.DataFrame(labels).duplicated(keep=False))[0]
    remove_idx = dup_idx[1 : 1 + len(dup_idx) // 2]
    handles = [h for i, h in enumerate(handles) if i not in remove_idx]
    labels = [l for i, l in enumerate(labels) if i not in remove_idx]
    ax_legend = fig.add_subplot(gs[-1, -1])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center", ncol=2, fontsize=24)

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
    t_all_params,
):
    """Compare CTF slope time courses for different parameters from spectral
    parameterization model."""
    all_params = [
        f.split("_")[-2]
        for f in os.listdir(params.SPARAM_DIR)
        if f.endswith(".fif")
    ]
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
            t_all_params,
            ctf_slopes_contrast=ctf_slopes_null_all_params,
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


def plot_ctf_slope_paired_ttest(
    ctf_slopes,
    t_arrays,
    t_window,
    ctf_slopes_shuffled=None,
    palette=None,
    save_fname=None,
    task_timings=params.TASK_TIMINGS,
    fig_dir=params.FIG_DIR,
):
    """Plot paired t-tests of channel tuning function (CTF) slope averaged
    time window for multiple parameters.
    """
    # Make empty list for CTF slope DataFrames
    ctf_slopes_dfs = []
    for i, (ctf_slope, t_arr) in enumerate(zip(ctf_slopes, t_arrays)):
        # If time window is None, use task timings
        if t_window == "delay":
            t_window = task_timings[i]

        # Average across time window
        t_window_idx = np.where(
            (t_arr >= t_window[0]) & (t_arr <= t_window[1])
        )[0]
        ctf_slope = -np.mean(ctf_slope[:, t_window_idx], axis=1)
        if ctf_slopes_shuffled is not None:
            ctf_slope_shuffled = -np.mean(
                ctf_slopes_shuffled[i][:, t_window_idx], axis=1
            )

        # Make DataFrame of CTF slopes
        ctf_slope_df = pd.DataFrame(
            ctf_slope, columns=["CTF slope"]
        ).reset_index(names="trial")
        ctf_slope_df.loc[:, "Shuffled location labels?"] = "No"
        ctf_slope_df.loc[:, "Task"] = str(i + 1)

        # Add DataFrame of CTF slopes for shuffled location labels if desired
        if ctf_slope_shuffled is not None:
            ctf_slope_shuffled_df = pd.DataFrame(
                ctf_slope_shuffled, columns=["CTF slope"]
            ).reset_index(names="trial")
            ctf_slope_shuffled_df.loc[:, "Shuffled location labels?"] = "Yes"
            ctf_slope_shuffled_df.loc[:, "Task"] = str(i + 1)
            ctf_slope_df = pd.concat(
                (ctf_slope_df, ctf_slope_shuffled_df), axis=0
            )

        # Add DataFrame to list
        ctf_slopes_dfs.append(ctf_slope_df)

    # Combine DataFrames of CTF slopes from each task into one big DataFrame
    ctf_slopes_big_df = pd.concat(ctf_slopes_dfs).reset_index()

    # Plot paired t-tests of CTF slopes for each task
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=ctf_slopes_big_df,
        x="Task",
        y="CTF slope",
        split=True,
        hue="Shuffled location labels?",
        palette=palette,
        inner="stick",
    )
    pairs = [
        ((str(task_num), "No"), (str(task_num), "Yes"))
        for task_num in list(set(ctf_slopes_big_df["Task"]))
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=ctf_slopes_big_df,
        x="Task",
        y="CTF slope",
        hue="Shuffled location labels?",
        plot="violinplot",
    )
    annotator.configure(
        test="t-test_paired",
        text_format="star",
        comparisons_correction="bonferroni",
    )
    annotator.apply_and_annotate()
    sns.despine(ax=ax)

    # Save figure
    if save_fname:
        os.makedirs(fig_dir, exist_ok=True)
        save_fname = f"{fig_dir}/{save_fname}"
        plt.savefig(save_fname, bbox_inches="tight", dpi=300)
    return
