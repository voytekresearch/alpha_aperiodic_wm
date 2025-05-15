# Import necessary modules
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import ttest_1samp, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from train_and_test_model import fit_model_desired_params
from fig2_analysis_pipeline import add_letter_labels
from fig4_compare_model_fits_across_tasks import zscore_model_fits
import params


def get_star_annotation(p):
    """Map p-values to significance markers."""
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def adjust_brightness(hex_color, factor):
    """
    Adjust the brightness of a HEX color.

    Parameters:
    - hex_color (str): The HEX color (e.g., "#RRGGBB").
    - factor (float): The brightness adjustment factor.
                      >1 to lighten, <1 to darken.

    Returns:
    - str: The adjusted HEX color.
    """
    rgb = mcolors.to_rgb(hex_color)  # Convert HEX to RGB
    adjusted_rgb = [
        min(max(c * factor, 0), 1) for c in rgb
    ]  # Adjust brightness
    return mcolors.to_hex(adjusted_rgb)  # Convert back to HEX


def compare_model_fits_across_windows(
    model_fits,
    t_arrays,
    params_to_plot=params.PARAMS_TO_PLOT,
    model_output_name="CTF slope",
    task_timings=params.TASK_TIMINGS,
    axes=None,
):
    """Plot model fits averaged over specified time windows with statistical
    annotations."""

    # Iterate over parameters and create subplots
    if axes is None:
        _, axes = plt.subplots(
            1, len(model_fits), figsize=(8 * len(model_fits), 6), sharey=True
        )
        if len(model_fits) == 1:  # Handle single subplot case
            axes = [axes]

    # Iterate over parameters and process model fits
    model_fits_all_params = []
    for ax, (param_name, model_fits_one_param), (param_name2, details) in zip(
        axes, model_fits.items(), params_to_plot.items()
    ):
        assert param_name == param_name2, "Mismatch in parameter names"
        t_arr = t_arrays[param_name]  # Get corresponding time array

        # Make empty list for DataFrames for this parameter
        model_fits_dfs = []

        # Process each task separately
        for task_num, model_fit_task in enumerate(model_fits_one_param):
            t_one_task = t_arr[task_num]

            # Create DataFrame for all time points for this task
            df = pd.DataFrame(model_fit_task, columns=t_one_task).melt(
                var_name="Time (s)", value_name=model_output_name
            )
            df["Task"] = str(task_num + 1)
            df["Parameter"] = details["name"]
            df["subject"] = df.groupby("Time (s)").cumcount()

            # Z-score the data before any averaging
            df, new_model_output_name = zscore_model_fits(
                df, model_output_name=model_output_name
            )

            # Define task-specific time windows
            current_task_timings = task_timings[task_num]
            time_windows = {
                "encoding": (0, current_task_timings[0]),
                "delay": current_task_timings,
            }

            # Iterate over time windows and average z-scored values
            for window_name, t_window in time_windows.items():
                # Define time window boundaries, replacing None with min/max
                t_start = (
                    t_window[0] if t_window[0] is not None else t_one_task[0]
                )
                t_end = (
                    t_window[1] if t_window[1] is not None else t_one_task[-1]
                )

                # Filter z-scored data for the time window
                window_data = df[
                    (df["Time (s)"] >= t_start) & (df["Time (s)"] <= t_end)
                ]

                # Compute mean z-scored values for this window
                avg_model_fit = (
                    window_data.groupby(["Parameter", "subject"])[
                        new_model_output_name
                    ]
                    .mean()
                    .reset_index()
                )
                avg_model_fit["Task"] = str(task_num + 1)
                avg_model_fit["Time Window"] = window_name

                # Append to the list
                model_fits_dfs.append(avg_model_fit)

        # Combine DataFrames for this parameter into one DataFrame
        model_fits_big_df = pd.concat(model_fits_dfs).reset_index(drop=True)

        # Generate palette for time windows
        base_color = details["color"]
        time_windows_list = ["encoding", "delay"]
        brightness_factors = [0.8, 1.2]  # Darker, original, lighter
        palette = {
            window: adjust_brightness(base_color, factor)
            for window, factor in zip(time_windows_list, brightness_factors)
        }

        # Plot using seaborn
        sns.stripplot(
            data=model_fits_big_df,
            x="Task",
            y=new_model_output_name,
            hue="Time Window",
            palette=palette,
            dodge=True,
            ax=ax,
        )
        ax.set_xlabel("Task", fontsize=18)
        ax.set_ylabel(new_model_output_name, fontsize=18)
        ax.tick_params(labelsize=14)
        ax.set_title(details["name"], fontsize=20)
        ax.legend(
            title="Time Window",
            loc="upper left",
            fontsize=14,
            title_fontsize=18,
        )
        sns.despine(ax=ax)

        # Compare encoding and delay time windows
        pairs = [
            ((task, "encoding"), (task, "delay"))
            for task in model_fits_big_df["Task"].unique()
        ]
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=model_fits_big_df,
            x="Task",
            y=new_model_output_name,
            hue="Time Window",
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            comparisons_correction="BH",
        )
        annotator.apply_and_annotate()

        # Perform one-sample t-tests against 0 and collect p-values
        p_values = []
        annotation_positions = []
        for task_num in model_fits_big_df["Task"].unique():
            for window in time_windows_list:
                window_data = model_fits_big_df[
                    (model_fits_big_df["Task"] == task_num)
                    & (model_fits_big_df["Time Window"] == window)
                ]
                _, p_val = ttest_1samp(window_data[new_model_output_name], 0)
                p_values.append(p_val)

                # Prepare annotation positions
                x_pos = (
                    float(task_num) - 0.2
                    if window == "encoding"
                    else float(task_num) + 0.2
                ) - 1
                y_pos = window_data[new_model_output_name].max() + 0.05
                annotation_positions.append((x_pos, y_pos))

        # Apply multiple comparisons correction
        adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

        # Add annotations with adjusted p-values
        for (x_pos, y_pos), adj_p_val in zip(
            annotation_positions, adjusted_p_values
        ):
            significance_label = get_star_annotation(adj_p_val)
            if significance_label == "ns":
                y_pos += 0.2
            print(significance_label, y_pos)
            ax.text(
                x_pos,
                y_pos,
                significance_label,
                ha="center",
                va="bottom",
                fontsize=12,
            )

        # Append to list of DataFrames for all parameters
        model_fits_all_params.append(model_fits_big_df)
    plt.tight_layout()
    return (
        pd.concat(model_fits_all_params).reset_index(drop=True),
        new_model_output_name,
    )


def compare_model_fits_across_params(
    model_fits_all_params,
    model_output_name,
    params_to_compare=(("Alpha total power", "Alpha oscillatory power"),),
    time_window="delay",
    axes=None,
    correction_method="fdr_bh",
):
    """
    Compare model fits across parameter pairs and include correlation coefficients and corrected p-values in the legend.

    Parameters:
        model_fits_all_params (pd.DataFrame): DataFrame with model fit data.
        model_output_name (str): Column name containing the output values for comparison.
        params_to_compare (list of tuples): Pairs of parameters to compare.
        save_fname (str, optional): File name to save the figure.
        fig_dir (str, optional): Directory to save the figure.
        time_window (str): Time window to filter the data.
        axes (list of Axes, optional): Existing matplotlib Axes for plotting.
        correction_method (str): Method for multiple comparison correction (default: 'fdr_bh').
    """
    # Filter for the desired time window
    model_fits_time_window = model_fits_all_params[
        model_fits_all_params["Time Window"] == time_window
    ]

    # Create subplots if not provided
    if axes is None:
        _, axes = plt.subplots(
            1,
            len(params_to_compare),
            figsize=(8 * len(params_to_compare), 6),
            squeeze=False,
        )
    if len(params_to_compare) == 1:  # Handle single subplot case
        axes = [axes]

    # Iterate over parameter pairs and create plots
    for i, param_pair in enumerate(params_to_compare):
        # Filter rows where Parameter matches the pair
        model_fits_pair = model_fits_time_window[
            model_fits_time_window["Parameter"].isin(param_pair)
        ]

        # Pivot to get x and y columns based on param_pair
        pivot_data = model_fits_pair.pivot(
            index=["subject", "Task"],  # Unique identifier for each point
            columns="Parameter",
            values=model_output_name,
        ).reset_index()

        # Compute correlation coefficients and p-values for each Task
        pivot_data = pivot_data.dropna(subset=[param_pair[0], param_pair[1]])
        tasks = pivot_data["Task"].unique()
        p_values = []

        for task in tasks:
            x = pivot_data[pivot_data["Task"] == task][param_pair[0]]
            y = pivot_data[pivot_data["Task"] == task][param_pair[1]]
            nom, denom = np.sum(x > y), len(x)
            print(
                f"Task {task}: {nom} of {denom} subjects "
                f"had higher {param_pair[0]} than {param_pair[1]}."
            )
            _, p = pearsonr(x, y)
            p_values.append(p)

        # Correct p-values for multiple comparisons
        corrected_p_values = multipletests(p_values, method=correction_method)[
            1
        ]

        # Get star annotations for corrected p-values
        star_annotations = [get_star_annotation(p) for p in corrected_p_values]

        # Add correlation and corrected p-value to the Task labels
        pivot_data["Task"] = pivot_data["Task"].apply(
            lambda t: f"{t}, {star_annotations[list(tasks).index(t)]}"
        )

        # Plot using seaborn
        sns.scatterplot(
            data=pivot_data,
            x=param_pair[0],
            y=param_pair[1],
            hue="Task",
            ax=axes[i],
            palette="Spectral",
        )
        axes[i].set_xlabel(f"{param_pair[0]} CTF slope", fontsize=18)
        axes[i].set_ylabel(f"{param_pair[1]} CTF slope", fontsize=18)
        axes[i].legend(
            title="Task", loc="upper left", fontsize=12, title_fontsize=18
        )
        sns.despine(ax=axes[i])

        # Plot identity line for comparison
        axes[i].plot(
            axes[i].get_xlim(),
            axes[i].get_ylim(),
            ls="--",
            c=".3",
            label="Identity line",
        )

    # Adjust layout and optionally save the figure
    plt.tight_layout()


def compare_model_fit_timecourses(
    fig_dir=params.FIG_DIR,
    save_fname="fig5_compare_model_fit_timecourses.pdf",
):
    """"""
    # Plot model fits across time windows
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )

    # Make gridspec
    rows = 2
    cols = (len(ctf_slopes) + 1) // 2
    fig = plt.figure(figsize=(7 * cols, 5 * rows))
    gs = fig.add_gridspec(rows, cols)

    # Plot model fits across time windows
    axes_windows = [
        fig.add_subplot(gs[row, col])
        for idx in range(len(ctf_slopes))
        for row, col in [divmod(idx, cols)]
    ]
    model_fits_df, output_name = compare_model_fits_across_windows(
        ctf_slopes,
        t_arrays,
        model_output_name="CTF slope",
        axes=axes_windows,
    )

    # Plot model fits across parameters
    ax_params = fig.add_subplot(gs[divmod(len(ctf_slopes), cols)])
    compare_model_fits_across_params(
        model_fits_df,
        output_name,
        axes=ax_params,
    )

    # Add letter labels
    axes_params = (
        [ax_params] if isinstance(ax_params, plt.Axes) else list(ax_params)
    )
    add_letter_labels(axes_windows + axes_params)

    # Save figure
    os.makedirs(fig_dir, exist_ok=True)
    fig_fname = f"{fig_dir}/{save_fname}"
    plt.savefig(fig_fname, bbox_inches="tight", dpi=300)
    return


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Set plot style
    plt.style.use(params.PLOT_SETTINGS)

    # Compare model fits across time windows and parameters
    compare_model_fit_timecourses()
