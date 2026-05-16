# Correcting Trend-Residual Covariance in Geostatistical Modeling Using Gram–Schmidt Orthogonalization

This repository contains the implementation code associated with the manuscript:

> **Correcting Trend-Residual Covariance in Geostatistical Modeling Using Gram–Schmidt Orthogonalization**

Code is provided as supplementary material to support reproducibility of all experimental results reported in the paper.

---

## Method Overview

Subsurface spatial data often exhibits non-stationary behavior driven by large-scale geological trends. Standard geostatistical workflows decompose observed values into a deterministic trend and a stochastic residual, assuming these components are uncorrelated and that their variances are additive. In practice, this orthogonality condition is rarely verified — and when violated, it deflates the residual variogram sill, causes simulation ensembles to underestimate true subsurface variability, and produces overconfident uncertainty intervals.

This work demonstrates that trend-residual covariance persists regardless of the trend modeling approach used, and proposes a Gram–Schmidt orthogonalization procedure to correct this coupling at sample locations. The correction is then extended across the full domain via simple kriging with a long-range variogram model.

### Core Correction Steps

1. **Trend modeling** — Estimate a large-scale trend surface (convolution or Gaussian Process Regression)
2. **Covariance diagnosis** — Quantify trend-residual cross-covariance at well locations
3. **Gram–Schmidt orthogonalization** — Project the residual onto the orthogonal complement of the trend, removing their covariance
4. **Spatial extension** — Krige the correction field to unsampled locations using a long-range variogram
5. **Impact assessment** — Evaluate effects on the residual variogram, kriging variance, SGS realizations, and uncertainty calibration

---

## Repository Structure

```
.
├── Practical_Covariance_Correction_Demo.ipynb   # End-to-end demonstration on a real 2D subsurface dataset
├── Experiment_Workflow_GP.ipynb                  # Controlled factorial experiment across synthetic datasets
├── Testing_Data_Generation.ipynb                 # Synthetic field generation and experimental design
├── auto_variogram_tuner.py                       # Automated variogram fitting pipeline (WLS, multi-model)
├── plot_variogram_reproduction.py                # Variogram reproduction diagnostics and plotting utilities
├── GS_experiment_datasets/                       # Generated synthetic datasets (factorial design)
│   ├── experiment_summary.csv                    # Index of all experimental cases
│   └── <case_id>_<trend>_<params>/              # Per-case folder: arrays, metadata, variogram figures
├── synthetic_trend_family.png                    # Visualization of trend families used in experiments
├── fig_diagnostics_scatter.png                   # Diagnostic scatter plots
└── requirements.txt                              # Python package dependencies
```

---

## Experimental Design

Synthetic property fields were constructed by combining a deterministic trend with a stochastic Gaussian residual field. The factorial design crossed the following parameters:

| Parameter | Symbol | Values | Description |
|---|---|---|---|
| Trend families | — | Channel, Asymmetric Ridge, Gaussian RF | Diverse large-scale structures |
| Range ratio | `rr` = range / L | 0.10, 0.15, 0.20, 0.30 | Residual spatial continuity relative to domain |
| Variance ratio | `vr` = Var(trend) / Var(Z) | 0.4, 0.5, 0.6, 0.7 | Trend signal strength |
| Variogram model | — | Spherical, Exponential | Residual covariance structure |
| Sampling design | `f_regular` | 1.0, 0.5, 0.25 | Fraction of uniformly placed wells (remainder preferential) |
| Sample size | N | 324 wells | Fixed per dataset |

Each dataset is stored in `GS_experiment_datasets/` with a `metadata.json` file, field arrays (`.npy`), sampled well data (`.csv`), and reconstruction figures.

---

## Implementation Details

### `Practical_Covariance_Correction_Demo.ipynb`

End-to-end walkthrough on a real 2D subsurface porosity dataset from [GeostatsGuy/GeoDataSets](https://github.com/GeostatsGuy/GeoDataSets). Covers:

- Data loading and visualization
- Trend modeling (convolution and Gaussian Process Regression)
- Gram–Schmidt covariance correction
- Residual variogram modeling (before and after correction)
- Effect on kriging local variance
- Effect on Sequential Gaussian Simulation ensembles and uncertainty calibration

### `Experiment_Workflow_GP.ipynb`

Batch experiment runner that applies the full correction and assessment pipeline across all synthetic datasets. Uses Optuna for automated variogram hyperparameter tuning and records before/after metrics across the factorial design.

### `Testing_Data_Generation.ipynb`

Constructs all synthetic fields using FFT-based spectral simulation and a two-stage sampling scheme (coarse regular grid + preferential acceptance). Generates the full `GS_experiment_datasets/` directory.

### `auto_variogram_tuner.py`

Fits spherical, exponential, and Gaussian variogram models to experimental variogram data using Cressie's weighted least-squares criterion. Returns the best-fit model as a `geostatspy`-compatible parameter set for use in sequential Gaussian simulation.

---

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy
- seaborn
- scikit-learn
- geostatspy
- optuna
- tqdm
- astropy

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Practical demonstration (recommended starting point)

Open and run `Practical_Covariance_Correction_Demo.ipynb` cell by cell. All parameters are set to the values used in the manuscript. The notebook downloads the sample dataset automatically from a public URL.

### Reproducing the factorial experiment

1. Run `Testing_Data_Generation.ipynb` to regenerate the synthetic datasets (or use the pre-generated `GS_experiment_datasets/` directory).
2. Run `Experiment_Workflow_GP.ipynb` to apply the correction workflow across all cases and collect results.

---

## Note to Reviewers

This code is provided for review purposes as supplementary material. All experimental results reported in the manuscript can be reproduced using this implementation. If you encounter any issues or have questions regarding the implementation, please contact the editorial office who can relay your questions to the authors while maintaining anonymity.
