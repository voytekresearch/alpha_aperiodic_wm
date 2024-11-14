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


def compare_model_fits_one_window(
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

    # Plot paired t-tests of model fits over averaged time window for multiple
    # parameters
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        verbose=False,
    )
    param_names = {
        "total_power": "Alpha total power",
        "linOscAUC": "Alpha oscillatory power",
        "exponent": "Aperiodic exponent",
    }
    compare_model_fits_one_window(
        ctf_slopes,
        t_arrays,
        t_window="delay",
        palette=cmr.take_cmap_colors("linear_blue_95_50_c20", 2),
        model_output_name="CTF slope",
        save_fname="model_fits_paired_ttest.png",
    )
