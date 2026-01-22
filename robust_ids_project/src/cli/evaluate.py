from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import transform
from src.eval.metrics import (
    compute_calibration_metrics,
    compute_classification_metrics,
)
from src.eval.plots import (
    plot_attack_rate_by_day,
    plot_drift_over_time,
    plot_performance_vs_day,
    plot_pr_curves,
    plot_reliability_curve,
    plot_roc_curves,
)
from src.models.drift import compute_drift, FeatureDriftResult
from src.utils.config import deep_get, ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate models under domain shift.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    random_seed = int(deep_get(cfg, ["project", "random_seed"], 42))
    set_global_seed(random_seed)

    paths_cfg: Dict[str, Any] = deep_get(cfg, ["paths"], {})
    robustness_cfg: Dict[str, Any] = deep_get(cfg, ["robustness"], {})
    eval_cfg: Dict[str, Any] = deep_get(cfg, ["eval"], {})

    artifacts_dir = ensure_dir(paths_cfg.get("artifacts_dir", "artifacts"))
    results_dir = ensure_dir(paths_cfg.get("results_dir", "results"))
    figures_dir = ensure_dir(results_dir / "figures")

    train_df = pd.read_csv(artifacts_dir / "train.csv")
    test_df = pd.read_csv(artifacts_dir / "test.csv")
    meta = load_yaml(artifacts_dir / "preprocess_meta.yaml")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")

    label_col = meta["label_col"]
    domain_col = meta["domain_col"]
    feature_cols: List[str] = meta["feature_cols"]

    models_dir = artifacts_dir / "models"

    plot_attack_rate_by_day(test_df, figures_dir / "attack_rate_by_day.png", domain_col=domain_col, label_col=label_col)

    logreg = joblib.load(models_dir / "logreg_base.joblib")
    rf = joblib.load(models_dir / "rf_base.joblib")

    metrics_rows: List[Dict[str, Any]] = []
    daily_metrics_rows: List[Dict[str, Any]] = []
    scores_for_plots: Dict[str, np.ndarray] = {}

    target_fpr = float(eval_cfg.get("recall_at_fpr", 0.01))
    low_fpr_targets = eval_cfg.get("low_fpr_targets")
    low_fpr_pauc_max = eval_cfg.get("low_fpr_pauc_max")

    def evaluate_model(name: str, model: Any, X: np.ndarray, y_true: np.ndarray, allow_abstain: bool = False) -> None:
        y_score = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        abstain_rate = None
        if allow_abstain:
            abstain_mask = y_pred == -1
            abstain_rate = float(abstain_mask.mean())
            y_pred = np.where(abstain_mask, 0, y_pred)

        m = compute_classification_metrics(
            y_true,
            y_pred,
            y_score,
            target_fpr=target_fpr,
            low_fpr_targets=low_fpr_targets,
            low_fpr_pauc_max=low_fpr_pauc_max,
        )
        m["model"] = name
        if abstain_rate is not None:
            m["abstain_rate"] = abstain_rate
        metrics_rows.append(m)
        scores_for_plots[name] = y_score

        return y_score, abstain_rate

    X_test = transform(preprocessor, test_df, feature_cols=feature_cols)
    y_test = test_df[label_col].to_numpy(dtype=int)

    logreg_score, _ = evaluate_model("logreg_base", logreg, X_test, y_test)
    rf_score, _ = evaluate_model("rf_base", rf, X_test, y_test)

    logreg_cal = None
    rf_cal = None
    recalib_cfg: Dict[str, Any] = robustness_cfg.get("recalibration", {})
    if recalib_cfg.get("enabled", True):
        try:
            logreg_cal = joblib.load(models_dir / "logreg_calibrated.joblib")
            logreg_cal_score, _ = evaluate_model("logreg_calibrated", logreg_cal, X_test, y_test)
            plot_reliability_curve(
                y_test,
                logreg_score,
                logreg_cal_score,
                figures_dir / "reliability_logreg.png",
                title="Reliability Curve: Logistic Regression",
            )
        except Exception:
            pass
        try:
            rf_cal = joblib.load(models_dir / "rf_calibrated.joblib")
            rf_cal_score, _ = evaluate_model("rf_calibrated", rf_cal, X_test, y_test)
            plot_reliability_curve(
                y_test,
                rf_score,
                rf_cal_score,
                figures_dir / "reliability_rf.png",
                title="Reliability Curve: Random Forest",
            )
        except Exception:
            pass

    try:
        logreg_conf = joblib.load(models_dir / "logreg_conformal.joblib")
        evaluate_model("logreg_conformal", logreg_conf, X_test, y_test, allow_abstain=True)
    except Exception:
        pass
    try:
        rf_conf = joblib.load(models_dir / "rf_conformal.joblib")
        evaluate_model("rf_conformal", rf_conf, X_test, y_test, allow_abstain=True)
    except Exception:
        pass

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)

    plot_roc_curves(y_test, scores_for_plots, figures_dir / "roc_curves.png")
    plot_pr_curves(y_test, scores_for_plots, figures_dir / "pr_curves.png")

    unique_domains = sorted(test_df[domain_col].unique())
    test_domains = meta["domain_split"]["test_domains"]
    train_domains = meta["domain_split"]["train_domains"]
    ref_df = train_df[train_df[domain_col].isin(train_domains[: len(train_domains) // 2])]

    drift_cfg: Dict[str, Any] = robustness_cfg.get("drift", {})
    drift_enabled = drift_cfg.get("enabled", True)
    method = drift_cfg.get("method", "both")
    psi_bins = int(drift_cfg.get("psi_bins", 10))
    psi_threshold = float(drift_cfg.get("psi_threshold", 0.2))
    alarm_min_fraction = float(drift_cfg.get("alarm_min_fraction_features", 0.1))

    daily_drift_rows: List[Dict[str, Any]] = []
    conformal_daily_rows: List[Dict[str, Any]] = []

    for day in unique_domains:
        day_mask = test_df[domain_col] == day
        X_day = transform(preprocessor, test_df[day_mask], feature_cols=feature_cols)
        y_day = test_df[day_mask][label_col].to_numpy(dtype=int)
        day_df = test_df[day_mask]

        for model_name in metrics_df["model"].unique():
            try:
                if "conformal" in model_name:
                    model = joblib.load(models_dir / f"{model_name.split('_')[0]}_{model_name.split('_')[1]}_conformal.joblib")
                elif "calibrated" in model_name:
                    model = joblib.load(models_dir / f"{model_name.split('_')[0]}_{model_name.split('_')[1]}_calibrated.joblib")
                else:
                    model = joblib.load(models_dir / f"{model_name.split('_')[0]}_{model_name.split('_')[1]}_base.joblib")
            except Exception:
                continue

            y_score_day = model.predict_proba(X_day)[:, 1]
            y_pred_day = model.predict(X_day)

            m = compute_classification_metrics(
                y_day,
                y_pred_day,
                y_score_day,
                target_fpr=target_fpr,
                low_fpr_targets=low_fpr_targets,
                low_fpr_pauc_max=low_fpr_pauc_max,
            )
            m["day"] = day
            m["model"] = model_name
            daily_metrics_rows.append(m)

            if "conformal" in model_name and hasattr(model, "predict_sets"):
                sets = model.predict_sets(X_day)
                coverage = float(np.mean([int(y_day[i] in s) for i, s in enumerate(sets)]))
                abstain_rate = float((y_pred_day == -1).mean())
                avg_set_size = float(np.mean([len(s) for s in sets]))
                conformal_daily_rows.append(
                    {
                        "day": day,
                        "model": model_name,
                        "coverage": coverage,
                        "abstain_rate": abstain_rate,
                        "avg_set_size": avg_set_size,
                    }
                )

        if drift_enabled:
            drift_results = compute_drift(ref_df, day_df, feature_cols=feature_cols, method=method, psi_bins=psi_bins)
            psi_values = [r.psi for r in drift_results.values() if r.psi is not None]
            max_psi = float(np.max(psi_values)) if psi_values else 0.0
            num_above_threshold = sum(1 for r in drift_results.values() if r.psi is not None and r.psi >= psi_threshold)
            alarm = num_above_threshold >= (alarm_min_fraction * len(drift_results))
            daily_drift_rows.append(
                {
                    "day": day,
                    "max_psi": max_psi,
                    "num_features_above_threshold": num_above_threshold,
                    "alarm": int(alarm),
                }
            )

    if daily_metrics_rows:
        daily_df = pd.DataFrame(daily_metrics_rows)
        daily_df.to_csv(results_dir / "daily_metrics.csv", index=False)
        for metric in ["roc_auc", "recall", "f1"]:
            if metric in daily_df.columns:
                plot_performance_vs_day(
                    daily_df,
                    figures_dir / f"performance_vs_day_{metric}.png",
                    domain_col="day",
                    metric_col=metric,
                    drift_alarms=pd.DataFrame(daily_drift_rows) if daily_drift_rows else None,
                )

    if daily_drift_rows:
        drift_daily_df = pd.DataFrame(daily_drift_rows)
        drift_daily_df.to_csv(results_dir / "drift_daily.csv", index=False)
        plot_drift_over_time(drift_daily_df, figures_dir / "drift_over_time.png", domain_col="day", score_col="max_psi")

    if conformal_daily_rows:
        conformal_daily_df = pd.DataFrame(conformal_daily_rows)
        conformal_daily_df.to_csv(results_dir / "conformal_daily.csv", index=False)

    if drift_enabled:
        drift_results = compute_drift(train_df, test_df, feature_cols=feature_cols, method=method, psi_bins=psi_bins)
        rows = []
        for feat, res in drift_results.items():
            rows.append(
                {
                    "feature": feat,
                    "ks_stat": res.ks_stat,
                    "ks_pvalue": res.ks_pvalue,
                    "psi": res.psi,
                }
            )
        drift_df = pd.DataFrame(rows).sort_values("psi", ascending=False)
        drift_df.to_csv(results_dir / "drift.csv", index=False)

    if recalib_cfg.get("enabled", True) and logreg_cal is not None:
        recent_val_df = pd.read_csv(artifacts_dir / "recent_val.csv")
        X_calib = transform(preprocessor, recent_val_df, feature_cols=feature_cols)
        y_calib = recent_val_df[label_col].to_numpy(dtype=int)
        cal_metrics_before = compute_calibration_metrics(y_calib, logreg.predict_proba(X_calib)[:, 1])
        cal_metrics_after = compute_calibration_metrics(y_calib, logreg_cal.predict_proba(X_calib)[:, 1])
        cal_metrics_df = pd.DataFrame(
            [
                {"model": "logreg", "stage": "before", **cal_metrics_before},
                {"model": "logreg", "stage": "after", **cal_metrics_after},
            ]
        )
        if rf_cal is not None:
            cal_metrics_rf_before = compute_calibration_metrics(y_calib, rf.predict_proba(X_calib)[:, 1])
            cal_metrics_rf_after = compute_calibration_metrics(y_calib, rf_cal.predict_proba(X_calib)[:, 1])
            cal_metrics_df = pd.concat(
                [
                    cal_metrics_df,
                    pd.DataFrame(
                        [
                            {"model": "rf", "stage": "before", **cal_metrics_rf_before},
                            {"model": "rf", "stage": "after", **cal_metrics_rf_after},
                        ]
                    ),
                ]
            )
        cal_metrics_df.to_csv(results_dir / "calibration_metrics.csv", index=False)


if __name__ == "__main__":
    main()

