# Robust Intrusion Detection Under Distribution Shift

A Python-based machine learning pipeline for intrusion detection that handles distribution shift through drift detection, probability calibration, and conformal prediction.

## Features

- Time-safe data splitting (forward-chaining)
- Statistical drift detection (KS test, PSI)
- Probability calibration (Platt scaling, isotonic regression)
- Conformal prediction with abstention
- Explainability analysis (permutation importance, SHAP)
- Low-FPR evaluation metrics

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Prepare your dataset (CSV format with timestamp, label, and feature columns)

2. Configure the experiment in `configs/experiment.yaml`

3. Run the pipeline:

```bash
python -m src.cli.preprocess --config configs/experiment.yaml
python -m src.cli.train --config configs/experiment.yaml
python -m src.cli.evaluate --config configs/experiment.yaml
python -m src.cli.explain --config configs/experiment.yaml
```

## Project Structure

```
src/
  data/          # Data loading, splitting, preprocessing
  models/        # Baseline models, drift detection, calibration, conformal
  eval/          # Metrics, plots, explainability
  cli/           # Command-line entrypoints
  utils/         # Configuration and utilities
configs/         # YAML configuration files
```

## Configuration

Edit `configs/experiment.yaml` to specify:
- Data paths and column names
- Model hyperparameters
- Robustness settings (drift, calibration, conformal)
- Evaluation metrics
- Output directories

## Output

Results are saved to:
- `artifacts/` - Models, preprocessors, split data
- `results/` - Metrics, drift analysis, explanations, figures

## Requirements

- Python 3.8+
- scikit-learn 1.5+
- pandas, numpy, matplotlib
- shap (optional, for tree model explanations)

