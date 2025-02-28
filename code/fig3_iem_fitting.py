# Import necessary modules
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
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
from fig2_analysis_pipeline import set_montage, add_letter_labels


def fit_iem_single_case(
    subj="CS_2",
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


def plot_stimulus(
    angle_deg,
    radius=0.55,
    circle_radius=0.15,
    color="#545454",
    bgcolor="#737373",
    ax=None,
):
    """
    Plots a circle on a square 'stimulus' at a given angle and radius
    around the center (0, 0).

    Parameters
    ----------
    angle_deg : float
        Angular position in degrees (0 to 360).
    radius : float
        The distance from the center (0, 0) at which to place the circle.
    circle_radius : float, optional
        The radius of the circle to be drawn. Defaults to 0.05.
    color : str, optional
        Color of the circle, e.g., 'black', 'red', etc. Defaults to 'black'.
    """

    # Convert from degrees to radians
    angle_rad = np.deg2rad(angle_deg)

    # Cartesian coordinates of the circle center
    x = radius * np.cos(angle_rad)
    y = radius * np.sin(angle_rad)

    # Create a figure and axis
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Plot a gray square "background" that is large enough for the circle
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(bgcolor)

    # Add the circle at (x, y)
    circle = Circle((x, y), circle_radius, facecolor=color, edgecolor="none")
    ax.add_patch(circle)

    # Add a cross at the center
    ax.plot([0, 0], [-0.03, 0.03], color=color, linewidth=2)
    ax.plot([-0.03, 0.03], [0, 0], color=color, linewidth=2)

    # Turn off axis ticks/labels if desired
    ax.set_xticks([])
    ax.set_yticks([])


def plot_channel_response(
    amps,
    ax=None,
    colors=None,
    channel_idx=None,
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
    for i, (bin, amp, c) in enumerate(zip(channel_centers, amps, colors)):
        alpha, linewidth = 0.5, 1
        if channel_idx is not None and i == channel_idx:
            alpha, linewidth = 1, 2
        ax.bar(
            bin,
            amp,
            width=bin_width,
            color=c,
            edgecolor="k",
            alpha=alpha,
            linewidth=linewidth,
        )
    ax.axis("off")
    return ax


def plot_eeg(
    epochs,
    data,
    ax=None,
):
    # Make figure
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    # Plot EEG
    epochs = set_montage(epochs)
    mne.viz.plot_topomap(data, epochs.info, axes=ax, show=False)
    return ax


def plot_channel_weights(
    epochs,
    weights,
    fig=None,
    ax=None,
    colors=None,
    channel_idx=None,
):
    # Make sure both fig and ax are either None or not None
    assert bool(fig) == bool(ax)

    # Get colors
    _, n_channels = weights.shape
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
        for i, (bin, w, c) in enumerate(zip(bins, weight, colors)):
            alpha, linewidth = 0.5, 1
            if channel_idx is not None and i == channel_idx:
                alpha, linewidth = 1, 2
            inset_ax.bar(
                bin,
                w,
                width=bins[1] - bins[0],
                color=c,
                edgecolor="k",
                alpha=alpha,
                linewidth=linewidth,
            )
        inset_ax.axis("off")  # Hide polar axes

        # Transform position to the main axis coordinates
        trans = main_ax.transData.transform((x, y))
        trans_inv = fig.transFigure.inverted().transform(trans)

        # Set the inset axis position
        inset_ax.set_position(
            [trans_inv[0] - 0.05, trans_inv[1] - 0.05, 0.1, 0.1]
        )

    # Place the main axis within the given axis
    if ax is not None:
        temp_path = "temp.png"
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
        return ax
    return main_ax


def plot_ctf_slope(
    amps,
    idx,
    ax=None,
    colors=None,
):
    # Get colors
    if colors is None:
        colors = [
            cc.cyclic_isoluminant[
                int(i * len(cc.cyclic_isoluminant) / len(amps))
            ]
            for i in range(len(amps))
        ]

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
    ax.plot(x, a * x + b, color="k", ls="--", lw=2)

    # Scatter plot
    ax.scatter(-dist_from_tuned, amps, c=colors, s=200)

    # Labels and styling
    ax.set_xlabel("Distance from tuned (degrees)", fontsize=20)
    ax.set_ylabel("Activation", fontsize=20)
    sns.despine(ax=ax)
    return ax


def make_iem_fitting_figure(
    fig_dir=params.FIG_DIR,
    save_fname="fig3_iem_fitting.png",
    iem_channel_train=1,
    iem_channel_test=5,
    title_fontsize=24,
):
    # Fit IEM model for single block
    iem, epochs, train_data, test_data = fit_iem_single_case()

    # Make figure
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 4)

    # Plot stimulus
    ax_stimulus_train = fig.add_subplot(gs[0, 0])
    plot_stimulus(
        IEM().channel_centers[iem_channel_train], ax=ax_stimulus_train
    )
    ax_stimulus_train.set_title(
        "Training stimulus", y=1.04, fontsize=title_fontsize
    )

    # Plot channel response
    crf = iem.design_matrix[:, iem_channel_train]
    ax_sim_response = fig.add_subplot(gs[0, 1], polar=True)
    plot_channel_response(
        crf, ax=ax_sim_response, channel_idx=iem_channel_train
    )
    ax_sim_response.set_title(
        "Predicted channel\nresponses ($C_{train}$)",
        y=0.99,
        fontsize=title_fontsize,
    )

    # Plot EEG for single training block
    ax_eeg_train = fig.add_subplot(gs[0, 2])
    eeg_train = np.mean(train_data[:, iem_channel_train :: len(crf)], axis=1)
    plot_eeg(epochs, eeg_train, ax=ax_eeg_train)
    ax_eeg_train.set_title(
        "Training EEG activity\n($B_{train}$)", y=0.98, fontsize=title_fontsize
    )

    # Plot channel weights
    ax_weights = fig.add_subplot(gs[0, 3])
    plot_channel_weights(
        epochs,
        iem.weights,
        ax=ax_weights,
        fig=fig,
        channel_idx=iem_channel_train,
    )
    ax_weights.set_title(
        "Estimated channel\nweights ($W$)", fontsize=title_fontsize
    )

    # Plot test stimulus
    ax_stimulus_test = fig.add_subplot(gs[1, 0])
    plot_stimulus(IEM().channel_centers[iem_channel_test], ax=ax_stimulus_test)
    ax_stimulus_test.set_title(
        "Test stimulus", y=1.04, fontsize=title_fontsize
    )

    # Plot inverted channel weights
    ax_inv_weights = fig.add_subplot(gs[1, 1])
    plot_channel_weights(
        epochs,
        iem.inv_weights.T,
        ax=ax_inv_weights,
        fig=fig,
        channel_idx=iem_channel_test,
    )
    ax_inv_weights.set_title(
        "Inverted channel\nweights ($W^{-1}$)", fontsize=title_fontsize
    )

    # Plot EEG for single testing block
    ax_eeg_test = fig.add_subplot(gs[1, 2])
    eeg_test = test_data[:, iem_channel_test]
    plot_eeg(epochs, eeg_test, ax=ax_eeg_test)
    ax_eeg_test.set_title(
        "Test EEG activity\n($B_{test}$)", y=0.98, fontsize=title_fontsize
    )

    # Plot predicted channel response
    ax_ctf = fig.add_subplot(gs[1, 3], polar=True)
    estimated_ctf = iem.estimated_ctfs[iem_channel_test, :]
    plot_channel_response(
        estimated_ctf, ax=ax_ctf, channel_idx=iem_channel_test
    )
    ax_ctf.set_title(
        "Estimated channel\nresponses ($C_{test}$)",
        y=0.98,
        fontsize=title_fontsize,
    )

    # Plot CTF slope
    ax_ctf_slope = fig.add_subplot(gs[2, 1:3])
    plot_ctf_slope(estimated_ctf, iem_channel_test, ax=ax_ctf_slope)
    ax_ctf_slope.set_title(
        "Channel tuning function (CTF) slope", fontsize=title_fontsize
    )

    # Add letter labels
    axes = fig.get_axes()
    add_letter_labels(axes)

    # Save figure
    if save_fname is not None:
        fig.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Set random seed
    np.random.seed(params.SEED)

    # Make figure
    make_iem_fitting_figure()
