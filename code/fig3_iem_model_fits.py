"""Functions to plot results from IEM analyses."""

# Import necessary modules
import cmasher as cmr
import numpy as np
import os
import pandas as pd
import pingouin as pg
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from train_and_test_model import fit_model_desired_params
import params


def plot_model_fit(
    model_fits,
    t_arrays,
    task_num,
    task_timings,
    param_names=None,
    model_output_name="CTF slope",
    palette=None,
    save_fname=None,
    plot_timings=True,
    plot_errorbars=False,
    ax=None,
):
    """Plot model fits across time for multiple parameters.

    Parameters
    ----------
    model_fits : dict
        Dictionary containing model fits for each parameter. Keys are parameter
        names and values are arrays containing model fits.
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
        param_names = {k: k for k in model_fits.keys()}

    # Make empty list for model fit DataFrames
    model_fits_dfs = []

    # Make DataFrame of model fits by time for each parameter
    for t_one_param, (param, model_fits_one_param) in zip(
        t_arrays.values(), model_fits.items()
    ):
        n = model_fits_one_param.shape[0]
        one_param_df = pd.DataFrame(model_fits_one_param, columns=t_one_param)
        one_param_df = one_param_df.melt(
            var_name="Time (s)",
            value_name=model_output_name,
        )
        one_param_df["Parameter"] = param_names[param]
        one_param_df["subject"] = one_param_df.groupby("Time (s)").cumcount()
        model_fits_dfs.append(one_param_df)

    # Combine DataFrames of model fits for each parameter into one big DataFrame
    model_fits_big_df = pd.concat(model_fits_dfs).reset_index(drop=True)

    # Z-score model fits using baseline period
    baseline = model_fits_big_df[model_fits_big_df["Time (s)"] < 0]
    baseline_stats = (
        baseline.groupby(["Parameter", "subject"])[model_output_name]
        .agg(["mean", "std"])
        .reset_index()
    )
    model_fits_big_df = model_fits_big_df.merge(
        baseline_stats, on=["Parameter", "subject"], suffixes=("", "_pre")
    )
    new_model_output_name = f"{model_output_name} (Z-scored on baseline)"
    model_fits_big_df[new_model_output_name] = (
        model_fits_big_df[model_output_name] - model_fits_big_df["mean"]
    ) / model_fits_big_df["std"]

    # Plot model fit time course for each parameter
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    ci = 95 if plot_errorbars else None
    ax = sns.lineplot(
        data=model_fits_big_df,
        hue="Parameter",
        x="Time (s)",
        y=new_model_output_name,
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
    ax.set_ylabel(new_model_output_name, size=28)
    ax.tick_params(labelsize=20)
    sns.despine(ax=ax)

    # Save if desired
    if save_fname:
        plt.savefig(save_fname, bbox_inches="tight", dpi=300)
        plt.close()


def plot_model_fit_time_courses(
    model_fits_all_params,
    t_all_params,
    param_sets=None,
    palettes=None,
    param_names=None,
    name="",
    title="",
    plt_errorbars=False,
    model_output_name="CTF slope",
    subjects_by_task=params.SUBJECTS_BY_TASK,
    fig_dir=params.FIG_DIR,
    task_timings=params.TASK_TIMINGS,
):
    """Plot model fit time courses for total power and parameters from spectral
    parameterization."""
    # Set default parameter sets, palettes, and parameter names
    if param_sets is None:
        param_sets = [list(model_fits_all_params.keys())]
    if palettes is None:
        palettes = ["rocket"]
    if param_names is None:
        param_names = {k: k for k in model_fits_all_params.keys()}

    # Create a GridSpec with one row and the number of tasks as columns
    num_tasks = len(subjects_by_task)
    fig = plt.figure(figsize=(48, 15), constrained_layout=True)
    gs = gridspec.GridSpec(2, num_tasks // 2 + 1, figure=fig)

    # Plot model fit time courses for parameters from spectral parameterization
    # model
    for task_num in range(len(subjects_by_task)):
        model_fits_one_task = {
            k: v[task_num] for k, v in model_fits_all_params.items()
        }
        t = {k: v[task_num] for k, v in t_all_params.items()}

        # Skip if no data for task
        if sum([len(v) for v in model_fits_one_task.values()]) == 0:
            continue

        # Get the corresponding subplot from the GridSpec
        ax = fig.add_subplot(gs[task_num % 2, task_num // 2])

        # Plot model fit time courses for each parameter set and palette
        for i, (param_set, palette) in enumerate(zip(param_sets, palettes)):
            # Get model fits for parameter set
            model_fits_one_param_set = {
                k: v for k, v in model_fits_one_task.items() if k in param_set
            }

            t_one_param_set = {k: v for k, v in t.items() if k in param_set}

            # Plot model fit time courses for parameter set and palette
            plt_timings = i == len(param_sets) - 1
            param_names_param_set = {k: param_names[k] for k in param_set}
            kwargs = {
                "palette": palette,
                "param_names": param_names_param_set,
                "plot_timings": plt_timings,
                "plot_errorbars": plt_errorbars,
                "ax": ax,
                "model_output_name": model_output_name,
            }
            plot_model_fit(
                model_fits_one_param_set,
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
    model_fits_fname = f"{fig_dir}/model_fits.png"
    if len(name) > 0:
        model_fits_fname = model_fits_fname.replace(".", f"_{name}.")
    plt.savefig(model_fits_fname, dpi=300, bbox_inches="tight")
    plt.close()


def plot_model_fit_paired_ttest(
    model_fits,
    t_arrays,
    t_window,
    model_fits_shuffled=None,
    palette=None,
    model_output_name="CTF slope",
    save_fname=None,
    task_timings=params.TASK_TIMINGS,
    fig_dir=params.FIG_DIR,
):
    """Plot paired t-tests of model fits over averaged time window for multiple
    parameters.
    """
    # Make empty list for model fit DataFrames
    model_fits_dfs = []
    for i, (model_fit, t_arr) in enumerate(zip(model_fits, t_arrays)):
        # If time window is None, use task timings
        if t_window == "delay":
            t_window = task_timings[i]

        # Average across time window
        t_window_idx = np.where(
            (t_arr >= t_window[0]) & (t_arr <= t_window[1])
        )[0]
        model_fit = np.mean(model_fit[:, t_window_idx], axis=1)
        if model_fits_shuffled is not None:
            model_fit_shuffled = np.mean(
                model_fits_shuffled[i][:, t_window_idx], axis=1
            )

        # Make DataFrame of model fits
        model_fit_df = pd.DataFrame(
            model_fit, columns=[model_output_name]
        ).reset_index(names="trial")
        model_fit_df.loc[:, "Shuffled location labels?"] = "No"
        model_fit_df.loc[:, "Task"] = str(i + 1)

        # Add DataFrame of model fits for shuffled location labels if desired
        if model_fit_shuffled is not None:
            model_fit_shuffled_df = pd.DataFrame(
                model_fit_shuffled, columns=[model_output_name]
            ).reset_index(names="trial")
            model_fit_shuffled_df.loc[:, "Shuffled location labels?"] = "Yes"
            model_fit_shuffled_df.loc[:, "Task"] = str(i + 1)
            model_fit_df = pd.concat(
                (model_fit_df, model_fit_shuffled_df), axis=0
            )

        # Add DataFrame to list
        model_fits_dfs.append(model_fit_df)

    # Combine DataFrames of model fits from each task into one big DataFrame
    model_fits_big_df = pd.concat(model_fits_dfs).reset_index(drop=True)

    # Plot paired t-tests of model fits for each task
    plt.figure(figsize=(10, 6))
    ax = sns.violinplot(
        data=model_fits_big_df,
        x="Task",
        y=model_output_name,
        split=True,
        hue="Shuffled location labels?",
        palette=palette,
        inner="stick",
    )
    pairs = [
        ((str(task_num), "No"), (str(task_num), "Yes"))
        for task_num in list(set(model_fits_big_df["Task"]))
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=model_fits_big_df,
        x="Task",
        y=model_output_name,
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


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    param_names = {
        "total_power": "Alpha total power",
        "linOscAUC": "Alpha oscillatory power",
        "exponent": "Aperiodic exponent",
    }
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        title="All parameters",
        palettes=["Set1"],
        param_names=param_names,
        plt_errorbars=True,
        name="iem",
    )
