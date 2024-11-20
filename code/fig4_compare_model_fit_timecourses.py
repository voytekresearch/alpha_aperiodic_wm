# Import necessary modules
import cmasher as cmr
import numpy as np
import os
import pandas as pd
import seaborn as sns
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
from train_and_test_model import fit_model_desired_params
import params


def compare_model_fits_across_windows(
    model_fits,
    t_arrays,
    palette=None,
    model_output_name="CTF slope",
    save_fname=None,
    task_timings=params.TASK_TIMINGS,
    fig_dir=params.FIG_DIR,
):
    """Plot model fits averaged over specified time windows with statistical annotations."""

    # Iterate over parameters and create subplots
    fig, axes = plt.subplots(
        1, len(model_fits), figsize=(8 * len(model_fits), 6), sharey=True
    )
    if len(model_fits) == 1:  # Handle single subplot case
        axes = [axes]

    for ax, (param_name, model_fits_one_param) in zip(
        axes, model_fits.items()
    ):
        t_arr = t_arrays[param_name]  # Get corresponding time array

        # Make empty list for DataFrames for this parameter
        model_fits_dfs = []

        # Process each task separately
        for task_num, model_fit_task in enumerate(model_fits_one_param):
            t_one_task = t_arr[task_num]

            # Define task-specific time windows
            current_task_timings = task_timings[task_num]
            time_windows = {
                "baseline": (None, 0),
                "stim": (0, current_task_timings[0]),
                "delay": current_task_timings,
            }

            # Iterate over time windows
            for window_name, t_window in time_windows.items():
                # Define time window boundaries, replacing None with min/max
                t_start = (
                    t_window[0] if t_window[0] is not None else t_one_task[0]
                )
                t_end = (
                    t_window[1] if t_window[1] is not None else t_one_task[-1]
                )

                # Compute indices for the time window
                t_window_idx = np.where(
                    (t_one_task >= t_start) & (t_one_task <= t_end)
                )[0]
                avg_model_fit = np.mean(
                    model_fit_task[:, t_window_idx], axis=1
                )

                # Create a DataFrame for this parameter and task
                df = pd.DataFrame(avg_model_fit, columns=[model_output_name])
                df["Task"] = str(task_num + 1)
                df["Time Window"] = window_name
                model_fits_dfs.append(df)

        # Combine DataFrames for this parameter into one DataFrame
        model_fits_big_df = pd.concat(model_fits_dfs).reset_index(drop=True)

        # Plot using seaborn
        sns.violinplot(
            data=model_fits_big_df,
            x="Task",
            y=model_output_name,
            hue="Time Window",
            palette=palette,
            inner="stick",
            ax=ax,
        )
        ax.set_title(param_name, fontsize=16)
        ax.legend(
            title="Time Window", bbox_to_anchor=(1.05, 1), loc="upper left"
        )

        # Statistical annotations: Compare "baseline" to other time windows globally
        pairs_set1 = [
            ((task, "baseline"), (task, "stim"))
            for task in model_fits_big_df["Task"].unique()
        ]
        pairs_set2 = [
            ((task, "baseline"), (task, "delay"))
            for task in model_fits_big_df["Task"].unique()
        ]
        pairs = pairs_set1 + pairs_set2
        annotator = Annotator(
            ax=ax,
            pairs=pairs,
            data=model_fits_big_df,
            x="Task",
            y=model_output_name,
            hue="Time Window",
        )
        annotator.configure(
            test="t-test_paired",
            text_format="star",
            comparisons_correction="bonferroni",
        )
        annotator.apply_and_annotate()

    plt.tight_layout()

    # Save the figure if needed
    if save_fname:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(f"{fig_dir}/{save_fname}", bbox_inches="tight", dpi=300)
    return


if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Plot model fits across time windows
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    compare_model_fits_across_windows(
        ctf_slopes,
        t_arrays,
        palette=sns.color_palette("Set1", 3),
        model_output_name="CTF slope",
        save_fname="fig4_compare_model_fit_timecourses.png",
    )
