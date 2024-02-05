"""Load data from MAT files acquired from Awh/Vogel lab, and convert to MNE 
Epochs objects for each subject."""

# Import necessary modules
import os
import os.path
import mne
from frozendict import frozendict
from pymatreader import read_mat
import numpy as np
import pandas as pd
import params


def _index_nested_object(nested, indices, subject, return_num_subjects=False):
    """Index nested Python object with tuple.

    Parameters:
    -----------
    nested : dict or list
        Nested Python object to index.
    indices : tuple
        Tuple of indices to index nested object with.
    subject : int
        Subject number to index with.
    return_num_subjects : bool (default: False)
        Whether to return number of subjects in nested object.

    Returns:
    --------
    nested : dict or list
        Nested Python object indexed with given indices.
    """
    # If given indices are not in fact indices, return them as the desired value
    if not isinstance(indices, tuple):
        return indices

    # Zoom in to next level
    if isinstance(nested, dict):
        if indices[0] not in nested.keys():
            return None
        nested = nested[indices[0]]
        indices = indices[1:]
    elif isinstance(nested, list):
        # Return number of subjects if desired
        if return_num_subjects:
            return len(nested)
        nested = nested[subject]

    # Return subject data if at end of nested indices
    if not indices:
        if isinstance(nested, list):
            # Return number of subjects if desired
            if return_num_subjects:
                return len(nested)
            if not isinstance(nested[0], str):
                nested = nested[subject]
        return nested

    # Continue with recursion if nested indices remain
    return _index_nested_object(nested, indices, subject, return_num_subjects)


def split_data_by_subject(
    experiment,
    experiment_vars=frozendict(params.EXPERIMENT_VARS),
    num_subjects=frozendict(params.NUM_SUBJECTS),
    download_dir=params.DOWNLOAD_DIR,
    epochs_dir=params.EPOCHS_DIR,
):
    """Split data for experiments into individual subjects.

    Parameters:
    -----------
    experiment : str
        Experiment to split data for.
    experiment_vars : dict (default: params.EXPERIMENT_VARS)
        Dictionary of experiment variables.
    num_subjects : dict (default: params.NUM_SUBJECTS)
        Dictionary of number of subjects for each experiment.
    download_dir : str (default: params.DOWNLOAD_DIR)
        Directory containing downloaded data.
    epochs_dir : str (default: params.EPOCHS_DIR)
        Directory to save Epochs data to.
    """
    # Make directory if necessary
    os.makedirs(epochs_dir, exist_ok=True)

    # Restrict experiment variables to selected experiment
    num_subjects = num_subjects[experiment]
    experiment_vars = experiment_vars[experiment]

    # Check if Epochs already generated for experiment to avoid loading large
    # MAT file if possible
    num_epo_files = len([f for f in os.listdir(epochs_dir) if experiment in f])
    if num_subjects == num_epo_files:
        return

    # Load data from experiment
    mat_fn = f"{download_dir}/spatialData_{experiment}.mat"
    exp_data = read_mat(mat_fn)

    # Convert data to Epochs for each subject
    for subject in range(num_subjects):
        # Skip if already done
        epochs_fname = f"{epochs_dir}/{experiment}_{subject}_eeg_epo.fif"
        if os.path.exists(epochs_fname):
            continue

        # Announce start of conversion
        print(
            f"Converting to Epochs for {experiment}_{subject}"
            f" {subject + 1}/{num_subjects}"
        )

        # Extract data from big nested dictionary
        eeg_data = _index_nested_object(
            exp_data, experiment_vars["data"], subject
        )

        # Move on to next subject if no data found
        if eeg_data is None:
            print("No EEG data")
            continue

        # Extract relevant experimental variables from big nested dictionary
        ch_labels = _index_nested_object(
            exp_data, experiment_vars["ch_labels"], subject
        )
        bad_trials = _index_nested_object(
            exp_data, experiment_vars["bad_trials"], subject
        ).astype(bool)
        sfreq = _index_nested_object(
            exp_data, experiment_vars["sfreq"], subject
        )
        pre_time = _index_nested_object(
            exp_data, experiment_vars["pre_time"], subject
        )
        post_time = _index_nested_object(
            exp_data, experiment_vars["post_time"], subject
        )
        pos_bin = _index_nested_object(
            exp_data, experiment_vars["pos_bin"], subject
        )
        bad_electrodes = _index_nested_object(
            exp_data, experiment_vars["bad_electrodes"], subject
        )
        art_pre_time = _index_nested_object(
            exp_data, experiment_vars["art_pre_time"], subject
        )
        art_post_time = _index_nested_object(
            exp_data, experiment_vars["art_post_time"], subject
        )
        error = _index_nested_object(
            exp_data, experiment_vars["error"], subject
        )
        rt = _index_nested_object(exp_data, experiment_vars["rt"], subject)

        # Create info, labeling bad electrodes
        info = mne.create_info(ch_labels, sfreq, ch_types="eeg", verbose=False)
        if bad_electrodes is None:
            bad_electrodes = []
        info["bads"] = [e for e in list(bad_electrodes) if e in ch_labels]

        # Create metadata DataFrame
        pos_bin_cleaned = pos_bin.flat
        error_cleaned = np.full(len(pos_bin_cleaned), np.nan)
        if error is not None:
            if len(error) > 1:
                error_cleaned = error.flat
        rt_cleaned = np.full(len(pos_bin_cleaned), np.nan)
        if rt is not None:
            if len(rt) > 1:
                rt_cleaned = rt.flat
        metadata_df = pd.DataFrame(
            {
                "pos_bin": pos_bin_cleaned,
                "error": error_cleaned,
                "rt": rt_cleaned,
            }
        )

        # Turn data array into MNE EpochsArray with proper cropping applied
        epochs = mne.EpochsArray(
            eeg_data,
            info,
            tmin=-np.abs(pre_time) / 1000,
            metadata=metadata_df,
            verbose=False,
        )
        if epochs.times[-1] != post_time / 1000:
            epochs = mne.EpochsArray(
                eeg_data,
                info,
                tmin=-np.abs(art_pre_time) / 1000,
                metadata=metadata_df,
                verbose=False,
            )
        else:
            # Crop data
            if art_pre_time:
                epochs.crop(
                    tmin=-np.abs(art_pre_time) / 1000,
                    tmax=art_post_time / 1000,
                )
                # Make sure times are consistent
                assert epochs.times[-1] == art_post_time / 1000

        # Drop bad trials
        epochs = epochs.drop(bad_trials, verbose=False)

        # Save epochs
        epochs.save(epochs_fname)


def split_data_all_subjects(download_dir=params.DOWNLOAD_DIR):
    # Split data by subjects
    experiments = set(
        [
            f.split(".")[-2].split("_")[-1]
            for f in os.listdir(download_dir)
            if ".mat" in f
        ]
    )
    for i, experiment in enumerate(sorted(experiments)):
        print(
            f"\nSplitting data for {experiment} ({i + 1}/{len(experiments)})"
        )
        split_data_by_subject(experiment)
    return


if __name__ == "__main__":
    split_data_all_subjects()
