# Import necessary modules
import fooof
from fooof.plts import plot_spectra
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mne
import seaborn as sns
import numpy as np
import params


def midpoint(pos1, pos2):
    return (np.array(pos1) + np.array(pos2)) / 2


def plot_sensors(
    epochs,
    ax=None,
    montage_name=params.JNP_MONTAGE,
    custom_channels=params.JNP_CUSTOM_CHANNELS,
):
    """Plot sensors for given epoched data."""
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

    # Set montage
    epochs.set_montage(montage, on_missing="ignore")

    # Plot sensors
    epochs.plot_sensors(show_names=True, show=False, axes=ax)
    return


def plot_epochs(
    epochs,
    ax,
    tp=None,
    chs_to_plot=params.CHANNELS_TO_PLOT,
    time_window_len=params.TIME_WINDOW_LEN,
):
    """Plot epoched data."""
    # Get channel data
    if chs_to_plot is None:
        chs_to_plot = epochs.ch_names
    ch_data = epochs.copy().apply_baseline().get_data(picks=chs_to_plot)

    # Get the spectral palette with different color for each channel
    colors = sns.color_palette("mako", len(chs_to_plot))

    # Plot epochs
    for i, (ch, c) in enumerate(zip(chs_to_plot, colors)):
        ax.plot(epochs.times, ch_data[0, i, :], color=c, label=ch)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
    ax.legend(title="Channel", loc="upper left")
    sns.despine(ax=ax)

    # Mark time point of interest if provided
    if tp is not None:
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        rect = Rectangle(
            (tp - time_window_len / 2, ax.get_ylim()[0] + 0.025 * yrange),
            time_window_len,
            0.975 * yrange,
            ec=(0, 0, 0, 0.5),
            fc=(0, 0, 0, 0.1),
            ls="--",
            lw=3,
        )
        ax.add_patch(rect)
    return


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
        dB=True,
        cmap="magma",
        vmin=np.min(trial_tfr),
    )
    # Mark time point of interest if provided
    if tp is not None:
        ax.axvline(tp, color=(0, 0, 0, 0.5), linestyle="--")
    return trial_tfr


def plot_sparam_psd(
    trial_tfr,
    ax,
    tp=0.5,
    n_peaks=params.N_PEAKS,
    peak_width_lims=params.PEAK_WIDTH_LIMS,
    fmin=params.FMIN,
    fmax=params.FMAX,
    alpha_band=params.ALPHA_BAND,
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

    # Plot spectral parameterization
    plot_spectra(freqs, powers, ax=ax, c="k")
    plot_spectra(
        freqs,
        10**fm._ap_fit,
        freq_range=(fmin, fmax),
        ax=ax,
        c="blue",
        ls="--",
    )
    ax.grid(False)
    sns.despine(ax=ax)
    ax.axvline(alpha_band[0], color="purple", linestyle="--", alpha=0.5)
    ax.axvline(alpha_band[1], color="purple", linestyle="--", alpha=0.5)
    ax.fill_between(
        freqs[low_freq_idx : high_freq_idx + 1],
        10 ** fm._ap_fit[low_freq_idx - 1 : high_freq_idx],
        y2=powers[low_freq_idx : high_freq_idx + 1],
        color="darkorange",
        alpha=0.2,
        label="Linear Oscillatory Alpha AUC",
    )
    ax.fill_between(
        freqs[low_freq_idx : high_freq_idx + 1],
        np.zeros(high_freq_idx - low_freq_idx + 1),
        y2=powers[low_freq_idx : high_freq_idx + 1],
        hatch="/",
        facecolor="w",
        edgecolor="darkorange",
        alpha=0.2,
        label="Linear Total Alpha AUC",
    )
    ax.legend(loc="upper right")
    return


def plot_sparam_params():
    return


def plot_analysis_pipeline(
    epochs_dir=params.EPOCHS_DIR,
    subj_id="CS_0",
    tp=0.5,
):
    """Plot entire analysis pipeline in one big figure."""
    # Load subject's EEG data
    epochs_fname = f"{epochs_dir}/{subj_id}_eeg_epo.fif"
    epochs = mne.read_epochs(epochs_fname, verbose=False)

    # Make gridspec
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, figure=fig)

    # Plot epochs
    ax_epochs = fig.add_subplot(gs[0, :2])
    plot_epochs(epochs, ax=ax_epochs, tp=tp)

    # Plot multitaper decomposition
    ax_multitaper = fig.add_subplot(gs[0, 2:])
    trial_tfr = plot_multitaper(epochs, ax=ax_multitaper, tp=tp)

    # Plot spectral parameterization PSD
    ax_sparam = fig.add_subplot(gs[1, 0])
    plot_sparam_psd(trial_tfr, ax=ax_sparam, tp=tp)

    # Plot spectral parameters across time
    ax_sparam_params = fig.add_subplot(gs[1, 1:-1])

    # TO-DO: Plot inverted encoding model

    # Save figure
    fig_fname = f"{params.FIG_DIR}/fig2_analysis_pipeline.png"
    fig.savefig(fig_fname, dpi=300, bbox_inches="tight")
    return


if __name__ == "__main__":
    # Plot analysis pipeline
    plt.style.use(params.PLOT_SETTINGS)
    plot_analysis_pipeline()
