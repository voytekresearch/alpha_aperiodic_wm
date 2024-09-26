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
    ch_data = epochs.copy().get_data(picks=chs_to_plot)

    # Get the spectral palette with different color for each channel
    colors = sns.color_palette("mako", len(chs_to_plot))

    # Plot epochs
    for i, (ch, c) in enumerate(zip(chs_to_plot, colors)):
        mean_norm = ch_data[0, i, :] - np.mean(ch_data[0, i, :])
        ax.plot(epochs.times, mean_norm, color=c, label=ch)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
    sns.despine(ax=ax)
    return


def plot_analysis_pipeline(
    epochs_dir=params.EPOCHS_DIR,
    subj_id="JNP_0",
):
    """Plot entire analysis pipeline in one big figure."""
    # Load subject's EEG data
    epochs_fname = f"{epochs_dir}/{subj_id}_eeg_epo.fif"
    epochs = mne.read_epochs(epochs_fname, verbose=False)

    # Make gridspec
    fig = plt.figure(figsize=(40, 24))
    gs = fig.add_gridspec(4, 3, figure=fig)

    # Plot sensors
    ax_sensors = fig.add_subplot(gs[0, 0])
    plot_sensors(epochs, ax=ax_sensors)

    # Plot epochs
    ax_epochs = fig.add_subplot(gs[0, 1:])
    plot_epochs(epochs, ax_epochs)

    # TO-DO: Plot multitaper decomposition

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
