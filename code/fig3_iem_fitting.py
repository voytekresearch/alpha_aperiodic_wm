# Import necessary modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
import colorcet as cc
import mne
from mne.channels.layout import _find_topomap_coords
import os
from iem import IEM
import params
from train_and_test_model import (
    load_param_data,
    equalize_param_data_across_trial_blocks,
)
from fig2_analysis_pipeline import set_montage


def fit_iem_single_case(
    subj="CS_1",
    param="linOscAUC",
    tp=0.5,
    param_dir=params.SPARAM_DIR,
):
    """
    Process a single block, single time point for one subject and parameter.

    Parameters
    ----------
    subj : str
        Subject ID.
    param : str
        Parameter to decode (e.g., 'alpha_power').
    param_dir : str
        Path to directory containing parameterized data.
    """
    # Load parameterized data
    epochs, times, param_data = load_param_data(
        subj,
        param,
        param_dir,
    )

    # Average parameterized data within trial blocks
    param_arr, _ = equalize_param_data_across_trial_blocks(
        epochs,
        times,
        param_data,
        average=True,
        distractors=False,
    )

    # Split into training and testing data
    train_data = np.delete(param_arr, 0, axis=1).reshape(
        param_arr.shape[0], -1, param_arr.shape[-1]
    )
    test_data = param_arr[:, 0, :, :]

    # Create labels for training and testing
    n_blocks = param_arr.shape[1]
    base_labels = IEM().channel_centers
    train_labels = np.tile(base_labels, n_blocks - 1)
    test_labels = base_labels

    # Select training and testing data from time point of interest
    tp_idx = np.argmin(np.abs(times - tp))
    train_data = train_data[:, :, tp_idx]
    test_data = test_data[:, :, tp_idx]

    # Train and test the IEM
    iem = IEM()
    iem.train_model(train_data, train_labels)
    iem.estimate_ctf(test_data, test_labels)
    iem.compute_ctf_slope()
    return iem, epochs, train_data, test_data


def plot_channel_response(
    amps,
    ax=None,
    colors=None,
    fig_dir=params.FIG_DIR,
    save_fname="channel_response.png",
):
    # Make figure
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Get colors
    if colors is None:
        colors = [
            cc.cyclic_isoluminant[
                int(i * len(cc.cyclic_isoluminant) / len(amps))
            ]
            for i in range(len(amps))
        ]

    # Make rose plot
    channel_centers = np.radians(IEM().channel_centers)
    bin_width = np.diff(channel_centers)[0]
    for bin, amp, c in zip(channel_centers, amps, colors):
        ax.bar(bin, amp, width=bin_width, color=c, edgecolor="k")
    ax.axis("off")
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return ax


def plot_eeg(
    epochs,
    data,
    ax=None,
    fig_dir=params.FIG_DIR,
    save_fname="eeg.png",
):
    # Make figure
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Plot EEG
    epochs = set_montage(epochs)
    mne.viz.plot_topomap(data, epochs.info, axes=ax, show=False)

    # Save figure
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return


def plot_channel_weights(
    epochs,
    weights,
    fig=None,
    ax=None,
    colors=None,
    fig_dir=params.FIG_DIR,
    save_fname="iem_channel_weights.png",
):
    # Make sure both fig and ax are either None or not None
    assert bool(fig) == bool(ax)

    # Get colors
    n_sensors, n_channels = weights.shape
    if colors is None:
        colors = [
            cc.cyclic_isoluminant[
                int(i * len(cc.cyclic_isoluminant) / n_channels)
            ]
            for i in range(n_channels)
        ]

    # Main scatter plot axis
    fig, main_ax = plt.subplots(figsize=(8, 8))

    # Set montage
    epochs = set_montage(epochs)

    # Plot sensors
    epochs.plot_sensors(
        show_names=False,
        show=False,
        axes=main_ax,
        sphere=(0.0, 0.02, 0.0, 0.095),
    )

    # Prepare topomap plot to get the transformed positions
    info = epochs.info
    picks = mne.pick_types(info, meg=False, eeg=True, exclude="bads")
    pos = _find_topomap_coords(
        info, picks=picks, sphere=(0.0, 0.02, 0.0, 0.095)
    )
    bins = np.radians(IEM().channel_centers)

    # Min-max normalize weights
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Add rose plots at the specified positions
    for (x, y), weight in zip(pos, weights):
        # Create a polar inset at the specific position
        inset_ax = fig.add_axes([0, 0, 0, 0], polar=True, label=f"{x},{y}")
        inset_ax.set_anchor("C")  # Center the polar plot

        # Plot the rose plot
        for bin, w, c in zip(bins, weight, colors):
            inset_ax.bar(
                bin,
                w,
                width=bins[1] - bins[0],
                color=c,
                edgecolor="k",
            )
        inset_ax.axis("off")  # Hide polar axes

        # Transform position to the main axis coordinates
        trans = main_ax.transData.transform((x, y))
        trans_inv = fig.transFigure.inverted().transform(trans)

        # Set the inset axis position
        inset_ax.set_position(
            [trans_inv[0] - 0.05, trans_inv[1] - 0.05, 0.1, 0.1]
        )

    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")

    # Place the main axis within the given axis
    if ax is not None:
        temp_path = f"{fig_dir}/temp.png"
        extent = main_ax.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        fig.savefig(temp_path, bbox_inches=extent)

        # Clear ax2 and display the saved image
        ax.clear()
        image = mpimg.imread(temp_path)
        ax.imshow(
            image,
            aspect="equal",
            extent=main_ax.get_xlim() + main_ax.get_ylim(),
        )
        ax.axis("off")
        os.remove(temp_path)
    return


def plot_ctf_slope(
    amps,
    ax=None,
    colors=None,
    fig_dir=params.FIG_DIR,
    save_fname="ctf_slope_scatter.png",
):
    # Get colors
    if colors is None:
        colors = [
            cc.cyclic_isoluminant[
                int(i * len(cc.cyclic_isoluminant) / len(amps))
            ]
            for i in range(len(amps))
        ]

    # Determine index of maximum amplitude
    idx = np.argmax(amps)

    # Calculate distance from tuned
    channel_centers = IEM().channel_centers
    feat_space_range = IEM().feat_space_range
    dist_from_tuned = np.min(
        [channel_centers, np.abs(channel_centers - feat_space_range)], axis=0
    )
    dist_from_tuned = np.roll(dist_from_tuned, idx)

    # Create figure and axis
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    # Fit and plot linear regression line
    a, b = np.polyfit(-dist_from_tuned, amps, 1)
    x = np.linspace(np.min(-dist_from_tuned), np.max(-dist_from_tuned), 100)
    ax.plot(x, a * x + b, color="k", ls="--")

    # Scatter plot
    ax.scatter(-dist_from_tuned, amps, c=colors)

    # Labels and styling
    ax.set_xlabel("Distance from tuned (degrees)", fontsize=14)
    ax.set_ylabel("Activation", fontsize=14)
    sns.despine(ax=ax)

    # Save figure
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return


def make_iem_fitting_figure(
    fig_dir=params.FIG_DIR,
    save_fname="fig3_iem_fitting.png",
    iem_channel_train=0,
    iem_channel_test=1,
):
    # Fit IEM model for single block
    iem, epochs, train_data, test_data = fit_iem_single_case()

    # Make figure
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(2, 5)

    # Plot channel response
    amps = iem.design_matrix[:, iem_channel_train]
    ax_sim_response = fig.add_subplot(gs[0, 1], polar=True)
    plot_channel_response(amps, ax=ax_sim_response)

    # Plot EEG for single training block
    ax_eeg_train = fig.add_subplot(gs[0, 2])
    eeg_train = np.mean(train_data[:, iem_channel_train :: len(amps)], axis=1)
    plot_eeg(epochs, eeg_train, ax=ax_eeg_train)

    # Plot channel weights
    ax_weights = fig.add_subplot(gs[0, 3])
    plot_channel_weights(epochs, iem.weights, ax=ax_weights, fig=fig)

    # Plot inverted channel weights
    ax_inv_weights = fig.add_subplot(gs[1, 1])
    plot_channel_weights(epochs, iem.inv_weights.T, ax=ax_inv_weights, fig=fig)

    # Plot EEG for single testing block
    ax_eeg_test = fig.add_subplot(gs[1, 2])
    eeg_test = test_data[:, iem_channel_test]
    plot_eeg(epochs, eeg_test, ax=ax_eeg_test)

    # Plot predicted channel response
    ax_ctf = fig.add_subplot(gs[1, 3], polar=True)
    estimated_ctf = iem.estimated_ctfs[iem_channel_test, :]
    plot_channel_response(estimated_ctf, ax=ax_ctf)

    # Plot CTF slope
    ax_slope = fig.add_subplot(gs[1, -1])
    plot_ctf_slope(estimated_ctf, ax=ax_slope)

    # Save figure
    if save_fname is not None:
        fig.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return


if __name__ == "__main__":
    # Set random seed
    np.random.seed(params.SEED)

    # Make figure
    make_iem_fitting_figure()
