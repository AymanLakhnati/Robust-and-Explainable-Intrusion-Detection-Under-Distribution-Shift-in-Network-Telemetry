from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import transform
from src.models.baselines import make_logistic_regression, make_random_forest
from src.models.conformal import ConformalAbstainer
from src.models.recalibration import ProbabilityCalibrator
from src.utils.config import deep_get, ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and robust models.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    random_seed = int(deep_get(cfg, ["project", "random_seed"], 42))
    set_global_seed(random_seed)

    paths_cfg: Dict[str, Any] = deep_get(cfg, ["paths"], {})
    models_cfg: Dict[str, Any] = deep_get(cfg, ["models"], {})
    robustness_cfg: Dict[str, Any] = deep_get(cfg, ["robustness"], {})

    artifacts_dir = ensure_dir(paths_cfg.get("artifacts_dir", "artifacts"))
    models_dir = ensure_dir(artifacts_dir / "models")

    train_df = pd.read_csv(artifacts_dir / "train.csv")
    recent_val_df = pd.read_csv(artifacts_dir / "recent_val.csv")
    meta = load_yaml(artifacts_dir / "preprocess_meta.yaml")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")

    label_col = meta["label_col"]
    feature_cols = meta["feature_cols"]

    X_train = transform(preprocessor, train_df, feature_cols=feature_cols)
    y_train = train_df[label_col].to_numpy(dtype=int)

    X_calib = transform(preprocessor, recent_val_df, feature_cols=feature_cols)
    y_calib = recent_val_df[label_col].to_numpy(dtype=int)

    logreg_params = models_cfg.get("logistic_regression", {})
    rf_params = models_cfg.get("random_forest", {})

    logreg = make_logistic_regression(logreg_params, random_seed=random_seed)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, models_dir / "logreg_base.joblib")

    rf = make_random_forest(rf_params, random_seed=random_seed)
    rf.fit(X_train, y_train)
    joblib.dump(rf, models_dir / "rf_base.joblib")

    recalib_cfg: Dict[str, Any] = robustness_cfg.get("recalibration", {})
    if recalib_cfg.get("enabled", True):
        method = recalib_cfg.get("method", "platt")

        logreg_cal = ProbabilityCalibrator(base_model=logreg, method=method)
        logreg_cal.fit(X_calib, y_calib)
        joblib.dump(logreg_cal, models_dir / "logreg_calibrated.joblib")

        rf_cal = ProbabilityCalibrator(base_model=rf, method=method)
        rf_cal.fit(X_calib, y_calib)
        joblib.dump(rf_cal, models_dir / "rf_calibrated.joblib")

    conformal_cfg: Dict[str, Any] = robustness_cfg.get("conformal", {})
    if conformal_cfg.get("enabled", True):
        alpha = float(conformal_cfg.get("alpha", 0.1))
        abstain_if_ambiguous = bool(conformal_cfg.get("abstain_if_ambiguous", True))

        try:
            logreg_for_conf = joblib.load(models_dir / "logreg_calibrated.joblib")
        except Exception:
            logreg_for_conf = logreg
        try:
            rf_for_conf = joblib.load(models_dir / "rf_calibrated.joblib")
        except Exception:
            rf_for_conf = rf

        logreg_conf = ConformalAbstainer(
            base_model=logreg_for_conf,
            alpha=alpha,
            abstain_if_ambiguous=abstain_if_ambiguous,
        ).fit(X_calib, y_calib)
        joblib.dump(logreg_conf, models_dir / "logreg_conformal.joblib")

        rf_conf = ConformalAbstainer(
            base_model=rf_for_conf,
            alpha=alpha,
            abstain_if_ambiguous=abstain_if_ambiguous,
        ).fit(X_calib, y_calib)
        joblib.dump(rf_conf, models_dir / "rf_conformal.joblib")


if __name__ == "__main__":
    main()

