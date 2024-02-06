"""Calculate parameters related to event-related potentials (ERPs)."""

# Import necessary modules
import os
import mne
import numpy as np
import time
import params


def calculate_erp_params_one_subject(
    subject, epochs_dir=params.EPOCHS_DIR, erp_dir=params.ERP_DIR
):
    """Calculate ERP parameters for one subject."""
    # Start timer
    start = time.time()

    # Load epochs
    epochs = mne.read_epochs(
        f"{epochs_dir}/{subject}_eeg_epo.fif", preload=True, verbose=False
    )

    # Baseline correction
    epochs = epochs.apply_baseline((None, 0), verbose=False)

    # Calculate N1 amplitude
    cropped = epochs.copy().crop(tmin=0.15, tmax=0.22).get_data(copy=True)
    n1_amp = np.min(cropped, axis=-1)

    # Convert to EpochsArray
    n1_epochs = mne.EpochsArray(
        n1_amp[:, :, np.newaxis],
        epochs.info,
        tmin=epochs.times[0],
        verbose=False,
    )

    # Save N1 amplitude
    save_fn = f"{erp_dir}/{subject}_n1amp_epo.fif"
    n1_epochs.save(save_fn)

    # Print processing time
    print(
        f"Calculated ERP parameters for {subject} in "
        f"{time.time() - start:.2f} seconds"
    )
    return


def calculate_erp_params_all_subjects(
    epochs_dir=params.EPOCHS_DIR, erp_dir=params.ERP_DIR
):
    """Calculate ERP parameters for all subjects."""
    # Create directory for ERP parameters if it doesn't exist
    os.makedirs(erp_dir, exist_ok=True)

    # Determine all subjects to calculate ERP parameters for
    subjects = ["_".join(f.split("_")[:2]) for f in os.listdir(epochs_dir)]
    subjects_processed = [
        "_".join(f.split("_")[:2]) for f in os.listdir(erp_dir)
    ]
    subjects_to_process = sorted(list(set(subjects) - set(subjects_processed)))

    # Calculate ERP parameters for each file
    for subject in subjects_to_process:
        calculate_erp_params_one_subject(
            subject, epochs_dir=epochs_dir, erp_dir=erp_dir
        )
    return


if __name__ == "__main__":
    calculate_erp_params_all_subjects()
