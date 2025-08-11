# Differential representations of spatial location by aperiodic and alpha oscillatory activity in working memory

This repository contains all the necessary code to reproduce the analysis and figures of the following manuscript:

**A. Bender, C. Zhao, E. Vogel, E. Awh, & B. Voytek. (2025). [Differential representations of spatial location by aperiodic and alpha oscillatory activity in working memory](https://www.pnas.org/doi/10.1073/pnas.2506418122). *Proc. Natl. Acad. Sci.*, 122(30) e2506418122, https://doi.org/10.1073/pnas.2506418122.**

## Datasets

The results are based on three publicly available datasets ([1](https://osf.io/bwzfj/), [2](https://osf.io/vw4uc/), [3](https://osf.io/47cmn/)) of electroencephalography (EEG) recordings from 112 participants across seven different working memory tasks:![](./figs/fig1_tasks.pdf)

The original results from these datasets are described in the following papers:

1. **Foster, J. J., Sutterer, D. W., Serences, J. T., Vogel, E. K., & Awh, E. (2016). [The topography of alpha-band activity tracks the content of spatial working memory](https://journals.physiology.org/doi/full/10.1152/jn.00860.2015). *Journal of neurophysiology*, 115(1), 168-177.**
2. **Foster, J. J., Bsales, E. M., Jaffe, R. J., & Awh, E. (2017). [Alpha-band activity reveals spontaneous representations of spatial position in visual working memory](https://www.cell.com/current-biology/fulltext/S0960-9822(17)31196-X). *Current biology*, 27(20), 3216-3223.**
3. **Sutterer, D. W., Foster, J. J., Adam, K. C., Vogel, E. K., & Awh, E. (2019). [Item-specific delay activity demonstrates concurrent storage of multiple active neural representations in working memory](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000239). *PLoS biology*, 17(4), e3000239.**

To reproduce our results, the EEG data for each of the 10 releases should be downloaded and placed in the directory specified by the `DOWNLOAD_DIR` variable in the `code/params.py` file. 

## Requirements

The provided Python 3 scripts require the following packages:
- `numpy` and `scipy` for numerical computation
- `pandas` for storage and manipulation of tabular data
- `mne` for reading and storing EEG data
- [`fooof`](https://fooof-tools.github.io/fooof/) for parameterizing neural power spectra
- `ray` for parallelizing the processing pipeline
- `frozendict` for immutable dictionaries
- `pymatreader` for reading MATLAB files
- `scikit-learn` for linear regression used in inverted encoding models
- `matplotlib`, `colorcet`, and `seaborn` for visualizing data and generating figures
- `pinguoin`, `astropy`, and `statannotations` for calculating and visualizing statistics

The particular versions used to perform the analysis and create the figures for the manuscript are specified in the `requirements.txt` file. To install all these packages in one's local Python environment, simplify enter `pip install -r requirements.txt` into the command line from the root of this GitHub repository.

## Processing Pipeline

Our processing pipeline can be run from start to finish by running the `code/proc_*.py` files in order. All parameters used to generate these output files are contained in the `code/params.py` file. To see how the results and figures change with different choices of parameters, change the parameters in the `code/params.py` file and rerun the necessary `code/proc_*.py` files.

*Warning*: The processing pipeline is computationally intensive (particularly the `code/proc1_spec_decomp_and_param.py` script) and took 2-3 days to run the entire pipeline on a cluster with 72 CPU cores.

## Figures

To generate figures from the manuscript, simply run the corresponding `code/fig*.py` file(s). Figures will be saved in the directory specified by the `FIG_DIR` variable in the `code/params.py` file.
