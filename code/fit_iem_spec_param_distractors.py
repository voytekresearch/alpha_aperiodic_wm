"""Fit inverted encoding model (IEM) for spatial location from alpha oscillatory 
 power and aperiodic exponent extracted from EEG data as Foster and colleagues
(https://pubmed.ncbi.nlm.nih.gov/26467522/."""

# Import necessary modules
import numpy as np
from train_and_test_model import fit_model_desired_params
import params

if __name__ == "__main__":
    # Set seed
    np.random.seed(params.SEED)

    # Fit IEM model with desired parameters on distractors
    ctf_slopes, t_arrays = fit_model_desired_params(
        sp_params=["total_power", "linOscAUC", "exponent"],
        distractors=True,
        verbose=False,
    )
