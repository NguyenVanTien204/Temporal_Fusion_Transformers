# Time Series with Python – Deep Learning Playbook

## Overview
This project is a hands-on exploration of modern deep learning techniques for retail time-series forecasting. Starting from Rossmann-style sales data, we build a reproducible workflow that moves from exploratory analysis to engineered features and ends with neural sequence models (e.g., LSTM) optimized for long-horizon store-level forecasts.

## Objectives
- Understand seasonality, promotion effects, and competitive signals through rigorous EDA.
- Standardize feature engineering pipelines (calendar, lag/rolling, competition/promo timing) to feed ML/DL models.
- Benchmark deep learning architectures for univariate and multivariate forecasting, with an initial focus on PyTorch-based LSTM baselines.
- Provide reusable code artifacts (feature tables, notebooks, training scripts) for rapid experimentation.

## Data Assets
- `data/train.csv`, `data/test.csv`, `data/store.csv`: raw Rossmann-like inputs.
- `data/model/train_fe.parquet`, `data/model/test_fe.parquet`: cleaned & feature-rich datasets exported from the EDA notebook.
- Generated artifacts: `data/model/lstm_best.pt` (best checkpoint) and `data/model/lstm_predictions.csv` (submission-ready forecasts).

## Repository Structure
```
├─ data/                # Raw CSVs + intermediate parquet files
├─ notebook/            # (Reserved) supplementary experimentation notebooks
├─ src/                 # Future Python modules for production pipelines
├─ EDA.ipynb            # 5-stage exploratory & feature engineering workflow
├─ LSTM.ipynb           # End-to-end LSTM training, validation, and inference
└─ requirements.txt     # Core Python dependencies
```

## Workflow
1. **EDA & Feature Engineering** (`EDA.ipynb`)
	- Cleans missing competition/promo data, enforces business sanity checks, and visualizes sales dynamics.
	- Creates standardized temporal, lag, rolling, and competition/promo-derived features.
	- Outputs parquet datasets for downstream modeling.
2. **Deep Learning Modeling** (`LSTM.ipynb`)
	- Loads engineered features, scales numerical columns, and builds per-store sliding windows.
	- Defines a multi-layer LSTM with dropout, trains using time-ordered splits, and tracks MAE/MSE.
	- Restores predictions to the original scale, evaluates validation accuracy, and exports next-day forecasts.

## Getting Started
```bash
pip install -r requirements.txt
```
1. Launch JupyterLab or VS Code notebooks.
2. Run `EDA.ipynb` sequentially to regenerate `train_fe.parquet` / `test_fe.parquet`.
3. Execute `LSTM.ipynb` to train the neural model and produce `lstm_predictions.csv`.

> **Tip:** GPU acceleration (CUDA) dramatically speeds up training; the notebook auto-detects available devices.

## Current Models & Experiments
- **Naïve Lag Baseline:** Time-based holdout with per-store `Lag_1` predictions (quick sanity metric).
- **LSTM Baseline:** 30-day window, 2-layer LSTM (hidden size 128) with AdamW + early stopping.

## Roadmap
1. Expand deep learning coverage (Temporal Convolutional Networks, N-BEATS, Transformer-based seq2seq).
2. Introduce automated hyperparameter search (Optuna) for sequence length, hidden dimensions, and learning rates.
3. Add rolling-origin evaluation utilities and experiment tracking (Weights & Biases/MLflow).
4. Productionize pipelines in `src/` with modular data loaders and model registries.

## Contributing
Pull requests and issue reports are welcome. Please describe the experiment, dataset version, and evaluation protocol when proposing changes.

## License
This project inherits the LICENSE file found in the repository root. Review it before using the code or datasets in external projects.
