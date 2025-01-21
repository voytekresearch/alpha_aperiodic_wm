"""Functions to plot results from IEM analyses."""

# Import necessary modules
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from train_and_test_model import fit_model_desired_params
from fig2_analysis_pipeline import add_letter_labels
import params


def zscore_model_fits(model_fits, model_output_name="CTF slope"):
    """Z-score model fits using baseline period."""
    # Compute mean and standard deviation of model output during baseline period
    baseline = model_fits[model_fits["Time (s)"] < 0]
    baseline_stats = (
        baseline.groupby(["Parameter", "subject"])[model_output_name]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Merge baseline statistics with model fits
    model_fits = model_fits.merge(
        baseline_stats, on=["Parameter", "subject"], suffixes=("", "_pre")
    )

    # Z-score model output using baseline statistics
    new_model_output_name = f"{model_output_name} (Z-scored on baseline)"
    model_fits[new_model_output_name] = (
        model_fits[model_output_name] - model_fits["mean"]
    ) / model_fits["std"]
    return model_fits, new_model_output_name


def plot_model_fit(
    model_fits,
    t_arrays,
    task_num,
    task_timings,
    sig_pval=0.05,
    params_to_plot=params.PARAMS_TO_PLOT,
    model_output_name="CTF slope",
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
    t_arrays : dict
        Dictionary containing time points for each parameter.
    task_num : int
        Task number.
    task_timings : list
        Timing of task events (e.g., onset, offset, etc.).
    params_to_plot : dict
        Dictionary containing metadata (e.g., color, label) for parameters to plot.
    model_output_name : str (default: "CTF slope")
        Name of the model output to display.
    save_fname : str (default: None)
        File name to save the figure.
    plot_timings : bool (default: True)
        Whether to plot task timings.
    plot_errorbars : bool (default: False)
        Whether to include error bars in the plot.

    ax : matplotlib.axes.Axes (default: None)
        Axes on which to plot.
    """
    # Extract parameter names and colors from params_to_plot
    param_names = {
        param: details["name"] for param, details in params_to_plot.items()
    }
    palette = {
        details["name"]: details["color"]
        for details in params_to_plot.values()
    }

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
    model_fits_big_df, new_model_output_name = zscore_model_fits(
        model_fits_big_df,
        model_output_name=model_output_name,
    )

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
    xmin = model_fits_big_df["Time (s)"].min()
    xmax = model_fits_big_df["Time (s)"].max()
    ax.set_xlim(xmin, xmax)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1))
    if task_num != 0 or not save_fname:
        legend.remove()
    _, _, _, ymax = ax.axis()

    # Plot task timings as shaded regions
    if plot_timings:
        # Shade the encoding period (stimulus onset to offset)
        ax.axvspan(
            0.0,
            task_timings[0],
            color="black",
            alpha=0.1,
            label="Encoding period",
        )

        # Shade the delay period (stimulus offset to the end of the plot)
        ax.axvspan(
            task_timings[0],
            task_timings[1],
            color="gray",
            alpha=0.1,
            label="Delay period",
        )

    # Set plot limits and aesthetics
    ax.set_xlim(None, task_timings[1])

    # Plot significance threshold
    if sig_pval:
        zscore = stats.norm.ppf(1 - sig_pval / 2)
        ax.axhline(zscore, c="r", ls="--", lw=2)
        ax.text(
            xmin + 0.05,
            zscore,
            r"$p < 0.05$",
            va="bottom",
            ha="left",
            size=24,
            color="r",
        )

    # Set plot title and labels
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
    params_to_plot=params.PARAMS_TO_PLOT,
    name="",
    title="",
    plt_errorbars=False,
    model_output_name="CTF slope",
    subjects_by_task=params.SUBJECTS_BY_TASK,
    fig_dir=params.FIG_DIR,
    task_timings=params.TASK_TIMINGS,
    save_fname="fig4_compare_model_fits_across_tasks.png",
):
    """Plot model fit time courses for total power and parameters from spectral
    parameterization.

    Parameters
    ----------
    model_fits_all_params : dict
        Dictionary containing model fits for all parameters across tasks.
    t_all_params : dict
        Dictionary containing time arrays for all parameters across tasks.
    param_sets : list of list (default: None)
        List of parameter sets to plot. If None, plots all parameters together.
    params_to_plot : dict (default: params.PARAMS_TO_PLOT)
        Dictionary containing metadata (e.g., color, label) for parameters to plot.
    name : str (default: "")
        Name to append to saved figure file.
    title : str (default: "")
        Title of the figure.
    plt_errorbars : bool (default: False)
        Whether to include error bars in the plots.
    model_output_name : str (default: "CTF slope")
        Name of the model output to display.
    subjects_by_task : dict
        Dictionary specifying subjects for each task.
    fig_dir : str
        Directory to save the figure.
    task_timings : list
        List of task timing intervals.
    """
    # Set default parameter sets
    if param_sets is None:
        param_sets = [list(model_fits_all_params.keys())]

    # Create a GridSpec with one row and the number of tasks as columns
    num_tasks = len(subjects_by_task)
    fig = plt.figure(figsize=(24, 36), constrained_layout=True)
    gs = gridspec.GridSpec(num_tasks // 2 + 1, 2, figure=fig)

    # Plot model fit time courses for parameters
    axes = []
    for task_num in range(len(subjects_by_task)):
        model_fits_one_task = {
            k: v[task_num] for k, v in model_fits_all_params.items()
        }
        t = {k: v[task_num] for k, v in t_all_params.items()}

        # Skip if no data for task
        if sum([len(v) for v in model_fits_one_task.values()]) == 0:
            continue

        # Get the corresponding subplot from the GridSpec
        ax = fig.add_subplot(gs[task_num // 2, task_num % 2])

        # Plot model fit time courses for each parameter set
        for i, param_set in enumerate(param_sets):
            # Get model fits for parameter set
            model_fits_one_param_set = {
                k: v for k, v in model_fits_one_task.items() if k in param_set
            }
            t_one_param_set = {k: v for k, v in t.items() if k in param_set}

            # Plot model fit time courses for parameter set
            plt_timings = i == len(param_sets) - 1
            plot_model_fit(
                model_fits_one_param_set,
                t_one_param_set,
                task_num,
                task_timings[task_num],
                params_to_plot=params_to_plot,
                plot_timings=plt_timings,
                plot_errorbars=plt_errorbars,
                ax=ax,
                model_output_name=model_output_name,
            )

            # Add axis to list of axes
            axes.append(ax)

    # Get legend handles and labels from the last axis
    handles, labels = ax.get_legend_handles_labels()
    dup_idx = np.where(pd.DataFrame(labels).duplicated(keep=False))[0]
    remove_idx = dup_idx[1 : 1 + len(dup_idx) // 2]
    handles = [h for i, h in enumerate(handles) if i not in remove_idx]
    labels = [l for i, l in enumerate(labels) if i not in remove_idx]
    ax_legend = fig.add_subplot(gs[-1, -1])
    ax_legend.axis("off")
    ax_legend.legend(handles, labels, loc="center", fontsize=36)

    # Add letter labels to subplots
    add_letter_labels(axes, size=64)

    # Save figure
    os.makedirs(fig_dir, exist_ok=True)
    model_fits_fname = f"{fig_dir}/{save_fname}"
    if len(name) > 0:
        model_fits_fname = model_fits_fname.replace(".", f"_{name}.")
    plt.savefig(model_fits_fname, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot CTF slope time courses for desired parameters from spectral
    # parameterization model
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    plot_model_fit_time_courses(
        ctf_slopes,
        t_arrays,
        title="All parameters",
        plt_errorbars=True,
        name="iem",
    )
