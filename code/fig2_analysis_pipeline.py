# Import necessary modules
import fooof
from fooof.plts import plot_spectra
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import numpy as np
import string
import params
from train_and_test_model import load_param_data


def midpoint(pos1, pos2):
    return (np.array(pos1) + np.array(pos2)) / 2


def set_montage(
    epochs,
    montage_name=params.JNP_MONTAGE,
    custom_channels=params.JNP_CUSTOM_CHANNELS,
):
    # Set standard montage and ignore missing channels
    montage = mne.channels.make_standard_montage(montage_name)

    # Compute custom channel positions
    custom_channel_pos = {
        ch: midpoint(
            montage.get_positions()["ch_pos"][ch1],
            montage.get_positions()["ch_pos"][ch2],
        )
        for ch, (ch1, ch2) in custom_channels.items()
    }

    # Add custom positions to the montage
    custom_montage = mne.channels.make_dig_montage(
        ch_pos=custom_channel_pos,
        coord_frame="head",  # Use the head coordinate frame
    )

    # Add custom montage to the standard montage
    montage = montage + custom_montage

    # Ignore EOG channels
    epochs.set_channel_types(
        {ch: "eog" for ch in epochs.ch_names if "EOG" in ch}
    )

    # Ignore stim channel
    epochs.set_channel_types(
        {ch: "misc" for ch in epochs.ch_names if "stim" in ch.lower()}
    )

    # Set montage
    epochs.set_montage(montage, on_missing="ignore")
    return epochs


def plot_epochs(
    epochs,
    gs_subplot,
    tp=None,
    trial_num=0,
    chs_to_plot=params.CHANNELS_TO_PLOT,
    time_window_len=params.TIME_WINDOW_LEN,
):
    """Plot epoched data for a specific trial with channels stacked vertically."""
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle

    # Get the data for the specified trial
    if trial_num >= len(epochs.events):
        raise ValueError(f"Trial number {trial_num} is out of range.")

    trial_epochs = epochs[trial_num].copy()

    # Check and filter valid channels
    chs_to_plot = [ch for ch in chs_to_plot if ch in trial_epochs.ch_names]
    if not chs_to_plot:
        print("No valid channels to plot.")
        return

    # Get channel data
    ch_data = trial_epochs.apply_baseline().get_data(picks=chs_to_plot)

    # Create a GridSpec for the provided SubplotSpec
    n_channels = len(chs_to_plot)
    gs = gridspec.GridSpecFromSubplotSpec(
        n_channels, 1, subplot_spec=gs_subplot, hspace=0.4
    )

    # Create individual axes for each channel
    axes = []
    for i, ch in enumerate(chs_to_plot):
        sub_ax = plt.subplot(gs[i])
        sub_ax.plot(trial_epochs.times, ch_data[0, i, :], color="black")
        sub_ax.set_ylabel(
            ch, rotation=0, labelpad=20, ha="right", va="center", fontsize=20
        )
        sub_ax.tick_params(
            axis="y", which="both", left=False
        )  # Remove y-axis ticks
        sub_ax.set_yticks([])  # No y-axis tick labels
        sns.despine(ax=sub_ax, left=True)

        # Add bounding box to the current axis
        if tp is not None and ch == chs_to_plot[0]:
            yrange = sub_ax.get_ylim()[1] - sub_ax.get_ylim()[0]
            rect = Rectangle(
                (
                    tp - time_window_len / 2,
                    sub_ax.get_ylim()[0] + 0.025 * yrange,
                ),
                time_window_len,
                0.975 * yrange,
                ec="black",
                fc="gray",
                alpha=0.2,
                ls="--",
                lw=2,
            )
            sub_ax.add_patch(rect)

        axes.append(sub_ax)

    # Label the x-axis only for the last subplot
    for sub_ax in axes[:-1]:
        sub_ax.set_xticklabels(
            []
        )  # Remove x-axis labels for all but the last subplot
    axes[-1].set_xlabel("Time (s)", fontsize=20)

    return axes


def plot_multitaper(
    epochs,
    ax,
    tp=None,
    ch=params.CHANNELS_TO_PLOT[0],
    trial_num=0,
    fmin=params.FMIN,
    fmax=params.FMAX,
    n_freqs=params.N_FREQS,
    time_window_len=params.TIME_WINDOW_LEN,
    decim_factor=params.DECIM_FACTOR,
):
    # Make frequencies linearly spaced
    freqs = np.linspace(fmin, fmax, n_freqs)

    # Make time window length consistent across frequencies
    n_cycles = freqs * time_window_len

    # Select data from just desired channel
    ch_epochs = epochs.pick(ch)

    # Get current trial of data
    trial = ch_epochs.copy().drop(
        [
            i
            for i in range(epochs.get_data(copy=True).shape[0])
            if i != trial_num
        ],
        verbose=False,
    )

    # Use multiple tapers to estimate spectrogram
    trial_tfr = mne.time_frequency.tfr_multitaper(
        trial,
        freqs,
        n_cycles,
        return_itc=False,
        picks="eeg",
        average=False,
        decim=decim_factor,
        verbose=False,
    )

    # Plot spectrogram
    trial_tfr[0].average().plot(
        axes=ax,
        show=False,
        cmap="magma",
        vmin=np.min(trial_tfr),
        colorbar=False,
    )
    im = ax.images[0]
    cbar = ax.get_figure().colorbar(im, ax=ax)
    cbar.set_label("Power", fontsize=20, rotation=270)
    label = cbar.ax.yaxis.label
    label.set_rotation_mode("anchor")
    cbar.ax.yaxis.set_label_coords(4.5, 0.5)

    # Mark time point of interest if provided
    if tp is not None:
        ax.axvline(tp, color="grey", linestyle="--")

    # Plot aesthetics
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("Frequency (Hz)", fontsize=20)
    return trial_tfr


def plot_sparam_psd(
    trial_tfr,
    ax,
    tp=0.5,
    n_peaks=params.N_PEAKS,
    peak_width_lims=params.PEAK_WIDTH_LIMS,
    fmin=params.FMIN,
    fmax=params.FMAX,
    plot_freq_range=params.PLOT_FREQ_RANGE,
    alpha_band=params.ALPHA_BAND,
    params_to_plot=params.PARAMS_TO_PLOT,
):
    """Plot spectral parameterization."""
    # Compute spectral parameterization
    freqs = trial_tfr.freqs
    tp_idx = np.argmin(np.abs(trial_tfr.times - tp))
    powers = trial_tfr.data[0, 0, :, tp_idx]
    fm = fooof.FOOOF(
        max_n_peaks=n_peaks, peak_width_limits=peak_width_lims, verbose=False
    )
    fm.fit(freqs, powers, freq_range=(fmin, fmax))

    # Determine indices for alpha band in frequencies array
    low_freq_idx = np.argmin(np.abs(freqs - alpha_band[0]))
    high_freq_idx = np.argmin(np.abs(freqs - alpha_band[1]))

    # Plot PSD
    plot_spectra(freqs, powers, ax=ax, c="k")

    # Plot exponent fit
    plot_spectra(
        freqs,
        10**fm._ap_fit,
        freq_range=plot_freq_range,
        ax=ax,
        c=params_to_plot["exponent"]["color"],
        ls="--",
    )

    # Plot AUC measure for band
    y_min, y_max = ax.get_ylim()
    ax.fill_between(
        freqs[low_freq_idx : high_freq_idx + 1],
        10 ** fm._ap_fit[low_freq_idx : high_freq_idx + 1],
        y2=powers[low_freq_idx : high_freq_idx + 1],
        color=params_to_plot["linOscAUC"]["color"],
        alpha=0.8,
        label=params_to_plot["linOscAUC"]["name"],
    )
    ax.fill_between(
        freqs[low_freq_idx : high_freq_idx + 1],
        0,
        hatch="/",
        y2=powers[low_freq_idx : high_freq_idx + 1],
        color=params_to_plot["total_power"]["color"],
        alpha=0.5,
        label=params_to_plot["total_power"]["name"],
    )

    # Plot greek character
    ax.text(
        (freqs[low_freq_idx] + freqs[high_freq_idx]) / 2,
        0.9 * y_max,
        rf'$\{params_to_plot["linOscAUC"]["name"].split()[0].lower()}$',
        fontsize=24,
        color=params_to_plot["linOscAUC"]["color"],
        va="top",
        ha="center",
    )

    # Plot bands
    ax.fill_betweenx(
        [0, y_max],
        freqs[low_freq_idx],
        freqs[high_freq_idx],
        facecolor=params_to_plot["total_power"]["color"],
        alpha=0.1,
    )

    # Plot aesthetics
    ax.set_ylim([0, y_max])
    ax.grid(False)
    ax.set_xlabel("Frequency (Hz)", fontsize=20)
    ax.set_ylabel("Power", fontsize=20)
    sns.despine(ax=ax)
    ax.legend(loc="upper right", fontsize=14)
    return


def plot_sparam_params(
    subj_id,
    ax,
    tp=None,
    trial_num=0,
    ch=params.CHANNELS_TO_PLOT[0],
    params_to_plot=params.PARAMS_TO_PLOT,
):
    """Plot spectral parameters across time."""
    for param_key, param in params_to_plot.items():
        # Load parameter data
        epochs, times, param_data = load_param_data(
            subj_id, param_key, param["dir"]
        )

        # Select data from the desired channel
        ch_data = param_data
        if ch is not None:
            ch_idx = epochs.ch_names.index(ch)
            ch_data = param_data[:, ch_idx, :]

        # Select data for the desired trial
        trial_data = ch_data
        if trial_num is not None:
            trial_data = ch_data[trial_num, :]

        # Z-score data
        trial_data = (trial_data - np.mean(trial_data)) / np.std(trial_data)

        # Plot parameter data
        ax.plot(times, trial_data, label=param["name"], color=param["color"])

    # Mark time point of interest if provided
    if tp is not None:
        ax.axvline(tp, color=(0, 0, 0, 0.5), linestyle="--")

    # Plot aesthetics
    ax.set_xlabel("Time (s)", fontsize=20)
    ax.set_ylabel("Z-score", fontsize=20)
    ax.legend(title="Parameter", title_fontsize=16, fontsize=12)
    sns.despine(ax=ax)
    return


def plot_sparam_topomaps(
    subj_id,
    fig,
    big_gs,
    tp=None,
    trial_num=0,
    params_to_plot=params.PARAMS_TO_PLOT,
):
    """Plot topomaps of spectral parameters at a given time point."""
    # Make gridspec
    gs = big_gs.subgridspec(len(params_to_plot), 1)

    for i, (param_key, param) in enumerate(params_to_plot.items()):
        # Load parameter data
        epochs, times, param_data = load_param_data(
            subj_id, param_key, param["dir"]
        )

        # Set montage
        epochs = set_montage(epochs)
        eeg_chs = ["eeg" in ch_type for ch_type in epochs.get_channel_types()]
        param_data = param_data[:, eeg_chs, :]

        # Select data from the desired trial
        trial_data = param_data
        if trial_num is not None:
            trial_data = param_data[trial_num, :]

        # Select data from the desired time point
        tp_idx = np.argmin(np.abs(times - tp))
        tp_data = trial_data[:, tp_idx]

        # Plot topomap
        ax = fig.add_subplot(gs[i])
        ax.set_title(param["name"], color=param["color"], fontsize=16)
        ax.set_aspect("equal")
        mne.viz.plot_topomap(
            np.real(tp_data),
            epochs.info,
            axes=ax,
            show=False,
            ch_type="eeg",
        )
    return


def letter_label(label, ax, x=-0.1, y=1.05, size=28):
    """Plot letter label on publication figures.

    Parameters
    ----------
    label: string
        Text to display as the bold label.
    ax : instance of Axes
        The axes to plot to.
    x : float | None
        x-position relative to axes left border.
    y : float | None
        y-position relative to axes top border.

    """
    # Plot the label
    ax.text(
        x,
        y,
        "%s" % label,
        transform=ax.transAxes,
        size=size,
        weight="bold",
    )


def add_letter_labels(axes, size=None):
    """Add letter labels to figure."""
    for i, ax in enumerate(axes):
        label = string.ascii_uppercase[i]
        kwargs = {} if size is None else {"size": size}
        letter_label(label, ax, **kwargs)
    return


def plot_analysis_pipeline(
    epochs_dir=params.EPOCHS_DIR,
    subj_id="CS_1",
    trial_num=20,
    tp=0.5,
    fig_dir=params.FIG_DIR,
):
    """Plot entire analysis pipeline in one big figure."""
    # Load subject's EEG data
    epochs_fname = f"{epochs_dir}/{subj_id}_eeg_epo.fif"
    epochs = mne.read_epochs(epochs_fname, verbose=False)

    # Make gridspec
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6, figure=fig)

    # Plot epochs directly using the grid space
    axes_epochs = plot_epochs(
        epochs.copy(), gs[0, :3], tp=tp, trial_num=trial_num
    )

    # Plot multitaper decomposition
    ax_multitaper = fig.add_subplot(gs[0, 3:])
    trial_tfr = plot_multitaper(
        epochs, ax=ax_multitaper, tp=tp, trial_num=trial_num
    )

    # Plot spectral parameterization PSD
    ax_sparam = fig.add_subplot(gs[1, :2])
    plot_sparam_psd(trial_tfr, ax=ax_sparam, tp=tp)

    # Plot spectral parameters across time
    ax_sparam_params = fig.add_subplot(gs[1, 2:-1])
    plot_sparam_params(
        subj_id, ax=ax_sparam_params, trial_num=trial_num, tp=tp
    )

    # Plot topomaps of spectral parameters for time point
    gs_topomaps = gs[1, -1]
    ax_topomaps = fig.add_subplot(gs_topomaps)
    ax_topomaps.axis("off")
    plot_sparam_topomaps(subj_id, fig, gs_topomaps, trial_num=trial_num, tp=tp)

    # Add letter labels
    axes = [axes_epochs[0]] + [  # First subplot in plot_epochs
        ax_multitaper,
        ax_sparam,
        ax_sparam_params,
        ax_topomaps,
    ]
    add_letter_labels(axes)

    # Save figure
    fig_fname = f"{fig_dir}/fig2_analysis_pipeline.png"
    fig.savefig(fig_fname, dpi=300, bbox_inches="tight")
    return


if __name__ == "__main__":
    # Plot analysis pipeline for the first 100 trials
    plt.style.use(params.PLOT_SETTINGS)
    plot_analysis_pipeline()
