# Import necessary modules
import matplotlib.pyplot as plt
from matplotlib import gridspec
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


def plot_epochs(epochs, ax=None, chs_to_plot=params.CHANNELS_TO_PLOT):
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
    sns.despine(ax=ax)
    return


def plot_multitaper(
    epochs,
    ax=None,
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
    return


def plot_analysis_pipeline(
    epochs_dir=params.EPOCHS_DIR,
    subj_id="CS_0",
):
    """Plot entire analysis pipeline in one big figure."""
    # Load subject's EEG data
    epochs_fname = f"{epochs_dir}/{subj_id}_eeg_epo.fif"
    epochs = mne.read_epochs(epochs_fname, verbose=False)

    # Make gridspec
    fig = plt.figure(figsize=(40, 24))
    gs = fig.add_gridspec(3, 4, figure=fig)

    # Plot epochs
    ax_epochs = fig.add_subplot(gs[0, :2])
    plot_epochs(epochs, ax=ax_epochs)

    # Plot multitaper decomposition
    ax_multitaper = fig.add_subplot(gs[0, 2:])
    plot_multitaper(epochs, ax=ax_multitaper)

    # TO-DO: Plot spectral parameterization

    # TO-DO: Plot inverted encoding model

    # Save figure
    fig_fname = f"{params.FIG_DIR}/fig2_analysis_pipeline.png"
    fig.savefig(fig_fname, dpi=300, bbox_inches="tight")
    return


if __name__ == "__main__":
    # Plot analysis pipeline
    plt.style.use(params.PLOT_SETTINGS)
    plot_analysis_pipeline()
