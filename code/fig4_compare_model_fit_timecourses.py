# Import necessary modules
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from train_and_test_model import fit_model_desired_params
from fig3_compare_model_fits_across_tasks import zscore_model_fits
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
    save_fname=None,
    task_timings=params.TASK_TIMINGS,
    fig_dir=params.FIG_DIR,
):
    """Plot model fits averaged over specified time windows with statistical
    annotations."""

    # Iterate over parameters and create subplots
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
        ax.set_title(details["name"], fontsize=16)
        ax.legend(title="Time Window", loc="upper left")
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
                y_pos = window_data[new_model_output_name].max() + 0.15
                annotation_positions.append((x_pos, y_pos))

        # Apply multiple comparisons correction
        adjusted_p_values = multipletests(p_values, method="fdr_bh")[1]

        # Add annotations with adjusted p-values
        for (x_pos, y_pos), adj_p_val in zip(
            annotation_positions, adjusted_p_values
        ):
            significance_label = get_star_annotation(adj_p_val)
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

    # Save the figure if needed
    if save_fname:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{save_fname}", bbox_inches="tight", dpi=300)
    return pd.concat(model_fits_all_params).reset_index(drop=True)


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot model fits across time windows
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    model_fits_df = compare_model_fits_across_windows(
        ctf_slopes,
        t_arrays,
        model_output_name="CTF slope",
        save_fname="fig4_compare_model_fit_timecourses.png",
    )
