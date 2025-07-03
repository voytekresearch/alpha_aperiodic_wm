#!/usr/bin/env python
"""
Compute Welch PSD for each subject and concatenate into a single NumPy array.
"""
# Import necessary libraries
import os
import mne
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import fooof
import warnings
from fooof.plts import plot_spectra
import params


def compute_subject_welch_psd(
    experiment,
    subject,
    epochs_dir=params.EPOCHS_DIR,
    fmin=params.FMIN,
    fmax=params.FMAX,
):
    """
    Compute Welch PSD for one subject, returning per-channel spectra.

    Returns
    -------
    freqs : ndarray
        Frequency bins.
    psd_chan : ndarray
        2D array shape (n_channels, n_freqs), PSD averaged over trials.
    """
    subj_id = f"{experiment}_{subject}"
    epochs_fname = os.path.join(epochs_dir, f"{subj_id}_eeg_epo.fif")
    if not os.path.exists(epochs_fname):
        raise FileNotFoundError(f"Epochs file not found: {epochs_fname}")

    # Load epochs
    epochs = mne.read_epochs(epochs_fname, verbose=False)

    # Determine window length (full epoch)
    n_per_seg = len(epochs.times)

    # Compute PSD via Epochs.compute_psd (Welch)
    psd = epochs.compute_psd(
        method="welch",
        fmin=fmin,
        fmax=fmax,
        picks="eeg",
        n_fft=n_per_seg,
        n_per_seg=n_per_seg,
        n_overlap=0,
        verbose=False,
    )
    freqs = psd.freqs
    data = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)

    # Average across trials only, keep channels dimension
    psd_chan = data.mean(axis=0)  # shape: (n_channels, n_freqs)
    return freqs, psd_chan


def compute_all_subjects_psd(
    epochs_dir=params.EPOCHS_DIR,
    subjects_by_task=params.SUBJECTS_BY_TASK,
    fmin=params.FMIN,
    fmax=params.FMAX,
):
    """
    Compute Welch PSD for all subjects across tasks, returning per-subject PSD arrays.

    Returns
    -------
    freqs_common : ndarray
        Frequency bins truncated to the shortest PSD length.
    psd_list : list of ndarray
        Each entry is a 2D array (n_channels, n_freqs) for one subject.
    subjects : list of tuples
        List of (experiment, subject) identifiers matching entries in psd_list.
    """
    # Gather all (experiment, subject) pairs
    subjects = []
    for exp, subj_ids in subjects_by_task:
        for subj in subj_ids:
            subjects.append((exp, subj))

    freqs_list = []
    psd_list = []
    for exp, subj in subjects:
        f, psd_chan = compute_subject_welch_psd(
            exp,
            subj,
            epochs_dir=epochs_dir,
            fmin=fmin,
            fmax=fmax,
        )
        freqs_list.append(f)
        psd_list.append(psd_chan)

    # Find shortest frequency vector length
    min_len = min(len(f) for f in freqs_list)
    freqs_common = freqs_list[0][:min_len]

    # Truncate each subject's PSD to that frequency length
    psd_truncated = [psd_chan[:, :min_len] for psd_chan in psd_list]

    return freqs_common, psd_truncated, subjects


def plot_one_psd(
    freqs,
    psd,
    ax,
    n_peaks=params.N_PEAKS,
    peak_width_lims=params.PEAK_WIDTH_LIMS,
    fmin=params.FMIN,
    fmax=params.FMAX,
    plot_freq_range=params.PLOT_FREQ_RANGE,
    alpha_band=params.ALPHA_BAND,
    params_to_plot=params.PARAMS_TO_PLOT,
):
    """
    Plot a single PSD vector with FOOOF overlays.

    Parameters
    ----------
    freqs : 1D array of frequencies
    psd : 1D array of power values (for one channel)
    ax : matplotlib Axes
    """
    # Fit FOOOF
    fm = fooof.FOOOF(
        max_n_peaks=n_peaks,
        peak_width_limits=peak_width_lims,
        verbose=False,
    )
    fm.fit(freqs, psd, freq_range=(fmin, fmax))

    lo = np.argmin(np.abs(freqs - alpha_band[0]))
    hi = np.argmin(np.abs(freqs - alpha_band[1]))
    y_max = psd.max() * 1.1

    # Raw PSD
    plot_spectra(freqs, psd, ax=ax, c="k")

    # Aperiodic fit
    ap_fit = 10**fm._ap_fit
    plot_spectra(
        freqs,
        ap_fit,
        freq_range=plot_freq_range,
        ax=ax,
        c=params_to_plot["exponent"]["color"],
        ls="--",
    )

    # Total-power AUC
    ax.fill_between(
        x=freqs[lo : hi + 1],
        y1=np.zeros_like(freqs[lo : hi + 1]),
        y2=psd[lo : hi + 1],
        hatch="/",
        color=params_to_plot["total_power"]["color"],
        alpha=0.5,
        label=params_to_plot["total_power"]["name"],
    )

    # Oscillatory AUC
    ax.fill_between(
        x=freqs[lo : hi + 1],
        y1=ap_fit[lo : hi + 1],
        y2=psd[lo : hi + 1],
        color=params_to_plot["linOscAUC"]["color"],
        alpha=0.8,
        label=params_to_plot["linOscAUC"]["name"],
    )

    # Greek label
    ax.text(
        (freqs[lo] + freqs[hi]) / 2,
        0.9 * y_max,
        rf"$\{params_to_plot['linOscAUC']['name'].split()[0].lower()}$",
        fontsize=16,
        color=params_to_plot["linOscAUC"]["color"],
        va="top",
        ha="center",
    )

    # Band shading
    ax.fill_betweenx(
        [0, y_max],
        freqs[lo],
        freqs[hi],
        facecolor=params_to_plot["total_power"]["color"],
        alpha=0.1,
    )

    # Cosmetics
    ax.set_xlim(plot_freq_range)
    ax.set_ylim([0, y_max])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.grid(False)
    sns.despine(ax=ax)
    ax.legend(fontsize=10)


def plot_all_subject_psds(
    freqs,
    psd_list,
    n_cols=4,
    fig_dir=params.FIG_DIR,
    **plot_kwargs,
):
    """
    Grid-plot the PSD of channel 0 for every subject.

    Parameters
    ----------
    freqs : 1D ndarray
        Frequency bins.
    psd_list : list of 2D arrays (n_channels, n_freqs)
    n_cols : int
        Number of columns in the grid.
    **plot_kwargs : dict
        Passed to plot_one_psd.
    """
    n_subjects = len(psd_list)
    n_rows = math.ceil(n_subjects / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 2.5),
        sharex=True,
        sharey=False,
    )
    axes = axes.flatten()

    for i, psd_chan in enumerate(psd_list):
        # Plot only channel 0 for each subject
        if psd_chan.shape[0] < 1:
            continue
        plot_one_psd(freqs, psd_chan[0, :], axes[i], **plot_kwargs)
        axes[i].set_title(f"Subject {i+1}, Ch 0", fontsize=12)

    # Turn off any extra axes
    for ax in axes[n_subjects:]:
        ax.axis("off")

    # Squeeze out extra padding at top
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Add a common title
    fig.suptitle("Welch PSD (Channel 0) for Each Subject", fontsize=16, y=1.02)

    # Save
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(
        os.path.join(fig_dir, "sfigX_welch_psd_each_subject.pdf"),
        bbox_inches="tight",
        dpi=300,
    )
    return


def detect_alpha_peaks(
    freqs,
    psd_list,
    n_peaks=params.N_PEAKS,
    peak_width_lims=params.PEAK_WIDTH_LIMS,
    fmin=params.FMIN,
    fmax=params.FMAX,
    freq_band=params.ALPHA_BAND,
):
    """
    Detect alpha peaks across all subject-channel spectra, tracking subject & channel ID.

    Returns
    -------
    peaks_df : pandas.DataFrame
        Columns: ['subject', 'channel', 'CF', 'Power', 'BW']
    """
    # Build a list of (subject_idx, channel_idx) for each spectrum
    idx_map = []  # will store (subject_index, channel_index)
    spec_list = []
    for subj_idx, psd_chan in enumerate(psd_list):
        n_chans = psd_chan.shape[0]
        for ch_idx in range(n_chans):
            spec_list.append(psd_chan[ch_idx, :])
            idx_map.append((subj_idx, ch_idx))
    data_2d = np.stack(spec_list, axis=0)  # shape (total_spectra, n_freqs)

    # Fit FOOOFGroup to all channels
    fg = fooof.FOOOFGroup(
        max_n_peaks=n_peaks,
        peak_width_limits=peak_width_lims,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fg.fit(freqs, data_2d, freq_range=(fmin, fmax))

    # Extract peaks DataFrame via to_df
    df_peaks = fg.to_df(fooof.Bands({"alpha": freq_band}))

    # Bring the first index level (group index) into a column
    df_peaks = df_peaks.reset_index()
    # After reset_index(), the first column is the old group index; rename it:
    grp_col = df_peaks.columns[0]
    df_peaks = df_peaks.rename(columns={grp_col: "GRP_index"})

    # Map each GRP_index back to (subject_idx, channel_idx)
    subjects_col = []
    channels_col = []
    for grp_idx in df_peaks["GRP_index"].values:
        subj_idx, ch_idx = idx_map[int(grp_idx)]
        subjects_col.append(subj_idx)
        channels_col.append(ch_idx)

    df_peaks["subject"] = subjects_col
    df_peaks["channel"] = channels_col
    return df_peaks


def plot_number_of_peaks_per_subject(peaks_df, fig_dir=params.FIG_DIR):
    """
    Plot histogram of number of non-NaN alpha power values (including zeros)
    per subject.
    """
    # Ensure the figure directory exists
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Count, for each subject, how many rows have a non-NaN alpha_pw
    counts = peaks_df.groupby("subject")["alpha_pw"].apply(
        lambda x: x.notna().sum()
    )

    # Choose bins so that each integer count is centered
    bins = np.arange(0, counts.max() + 2) - 0.5

    # Plot histogram of those counts
    sns.ecdfplot(
        x=counts,
        stat="proportion",
        complementary=False,
    )
    plt.xlabel("Number of Channels with Alpha Peak in PSD")
    plt.ylabel("Proportion of Participants")

    # Save the figure
    sns.despine()
    plt.savefig(
        f"{fig_dir}/sfigX_alpha_pw_counts_per_subject.pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    # Compute PSDs for all subjects
    freqs, psd_list, subjects = compute_all_subjects_psd()
    print(
        f"Computed PSD for {len(subjects)} subjects; data list length: {len(psd_list)}"
    )

    # Detect alpha peaks across all channels
    peaks_df = detect_alpha_peaks(freqs, psd_list)

    # Plot histogram of number of peaks per subject
    plot_number_of_peaks_per_subject(peaks_df)
