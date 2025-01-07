# Import necessary modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import colorcet as cc
import params


def simulate_channel_response(
    colors,
    ax=None,
    fig_dir=params.FIG_DIR,
    save_fname="simulated_iem_channel.png",
):
    # Make figure
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Define bins and amplitudes
    bins = np.linspace(np.pi / 2, -3 / 2 * np.pi, 9)[:-1]
    amps = [0.5, 1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Make rose plot
    for bin, amp, c in zip(bins, amps, colors):
        ax.bar(bin, amp, width=bins[1] - bins[0], color=c, edgecolor="k")
    ax.axis("off")
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return ax


def plot_channel_weights(
    colors,
    fig_dir=params.FIG_DIR,
    save_fname="iem_channel_weights.png",
):
    # Make figure
    n_rows, n_columns = 4, 3
    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    gs = fig.add_gridspec(n_rows, n_columns)

    # Define bins and channel weights
    bins = np.linspace(np.pi / 2, -3 / 2 * np.pi, len(colors) + 1)[:-1]
    weights = np.random.rand(n_rows * n_columns, len(colors))

    # Make rose plot for each set of weights
    axes = []
    for i in range(n_rows * n_columns):
        ax = fig.add_subplot(gs[i], polar=True)
        for bin, weight, c in zip(bins, weights[i, :], colors):
            ax.bar(
                bin,
                weight,
                width=bins[1] - bins[0],
                color=c,
                bottom=0.8,
                edgecolor="k",
            )
        ax.axis("off")
        axes.append(ax)
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return weights, axes


def plot_inverted_channel_weights(
    weights,
    colors,
    fig_dir=params.FIG_DIR,
    save_fname="iem_channel_inverse.png",
):
    # Make figure
    n_rows, n_columns = 4, 3
    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    gs = fig.add_gridspec(n_rows, n_columns)

    # Define bins and calculate inverse for channel weights
    bins = np.linspace(np.pi / 2, -3 / 2 * np.pi, len(colors) + 1)[:-1]
    weights_inv = np.linalg.inv(weights.T @ weights) @ weights.T
    weights_inv = (weights_inv - np.min(weights_inv)) / (
        np.max(weights_inv) - np.min(weights_inv)
    )

    # Make rose plot for each set of inverted weights
    axes = []
    for i in range(n_rows * n_columns):
        ax = fig.add_subplot(gs[i], polar=True)
        for bin, weight_inv, c in zip(bins, weights_inv[:, i], colors):
            ax.bar(
                bin,
                weight_inv,
                width=bins[1] - bins[0],
                color=c,
                bottom=0.8,
                edgecolor="k",
            )
        ax.axis("off")
        axes.append(ax)
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return axes


def plot_channel_response(
    colors,
    ax=None,
    fig_dir=params.FIG_DIR,
    save_fname="predicted_channel_response.png",
):
    # Make figure
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)

    # Define bins and amplitudes
    bins = np.linspace(np.pi / 2, -3 / 2 * np.pi, 9)[:-1]
    amps = [0.021, 0.21, 0.11, 0.0, 0.05, 0.4, 0.8, 0.5]

    # Make rose plot
    for bin, amp, c in zip(bins, amps, colors):
        ax.bar(bin, amp, width=bins[1] - bins[0], color=c, edgecolor="k")
    ax.axis("off")
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return ax, amps


def plot_ctf_slope(
    colors,
    dist_from_tuned,
    amps,
    fig_dir=params.FIG_DIR,
    save_fname="ctf_slope_scatter.png",
):
    idx = np.argmax(amps)
    dist_from_tuned = np.roll(dist_from_tuned, idx)

    # Plot CTF slope
    fig = plt.figure(figsize=(4, 4))
    a, b = np.polyfit(-dist_from_tuned, amps, 1)
    x = np.linspace(np.min(-dist_from_tuned), np.max(-dist_from_tuned), 100)
    plt.plot(x, a * x + b, color="k", ls="--")
    plt.scatter(-dist_from_tuned, amps, c=colors)
    plt.xlabel("Distance from tuned (degrees)", fontsize=14)
    plt.ylabel("Activation", fontsize=14)
    sns.despine()
    if save_fname is not None:
        plt.savefig(f"{fig_dir}/{save_fname}", dpi=300, bbox_inches="tight")
    return


def make_iem_fitting_figure(
    n_channels=params.IEM_N_CHANNELS,
    feat_space_edges=params.IEM_FEAT_SPACE_EDGES,
    basis_func=params.IEM_BASIS_FUNC,
):
    # Create basis set
    feat_space_range = (np.diff(feat_space_edges)[0] + 1).astype(int)
    theta = np.linspace(*feat_space_edges, feat_space_range)
    channel_centers = np.linspace(
        feat_space_edges[0],
        feat_space_edges[-1] + 1,
        n_channels + 1,
    )[:-1].astype(int)
    basis_set = np.array(
        [
            np.roll(
                basis_func(theta),
                channel_center - len(theta) // 2,
            )
            for channel_center in channel_centers
        ]
    )
    dist_from_tuned = np.min(
        [channel_centers, np.abs(channel_centers - feat_space_range)], axis=0
    )

    # Get colors
    colors = [
        cc.cyclic_isoluminant[int(i * len(cc.cyclic_isoluminant) / n_channels)]
        for i in range(n_channels)
    ]

    # Plot channel response
    simulate_channel_response(colors)

    # Plot channel weights
    weights, _ = plot_channel_weights(colors)

    # Plot inverted channel weights
    plot_inverted_channel_weights(weights, colors)

    # Plot predicted channel response
    _, amps = plot_channel_response(colors)

    # Plot CTF slope
    plot_ctf_slope(colors, dist_from_tuned, amps)
    return


if __name__ == "__main__":
    make_iem_fitting_figure()
