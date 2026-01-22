from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

from src.eval.metrics import reliability_curve


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_roc_curves(
    y_true: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 6))
    for name, scores in scores_by_model.items():
        RocCurveDisplay.from_predictions(y_true, scores, name=name)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curves(
    y_true: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 6))
    for name, scores in scores_by_model.items():
        PrecisionRecallDisplay.from_predictions(y_true, scores, name=name)
    baseline = y_true.mean()
    plt.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(
    features: List[str],
    importances: np.ndarray,
    out_path: Path,
    title: str = "Permutation Importance",
    top_k: int = 20,
) -> None:
    _ensure_dir(out_path.parent)
    idx = np.argsort(importances)[::-1][:top_k]
    imp = importances[idx]
    feats = [features[i] for i in idx]

    plt.figure(figsize=(8, max(4, 0.3 * len(feats))))
    y_pos = np.arange(len(feats))
    plt.barh(y_pos, imp[::-1])
    plt.yticks(y_pos, feats[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray | None,
    out_path: Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability Curve",
) -> None:
    _ensure_dir(out_path.parent)
    prob_pred_before, prob_true_before = reliability_curve(y_true, y_prob_before, n_bins=n_bins)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred_before, prob_true_before, "o-", label="Before calibration", linewidth=2)
    if y_prob_after is not None:
        prob_pred_after, prob_true_after = reliability_curve(y_true, y_prob_after, n_bins=n_bins)
        plt.plot(prob_pred_after, prob_true_after, "s-", label="After calibration", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_drift_over_time(
    drift_df: pd.DataFrame,
    out_path: Path,
    *,
    domain_col: str = "day",
    score_col: str = "max_psi",
) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 5))
    days = drift_df[domain_col].values
    scores = drift_df[score_col].values
    alarms = drift_df.get("alarm", np.zeros(len(drift_df))).values
    plt.plot(days, scores, "o-", linewidth=2, markersize=5, label="Drift score")
    if np.any(alarms):
        alarm_days = days[alarms.astype(bool)]
        alarm_scores = scores[alarms.astype(bool)]
        plt.scatter(alarm_days, alarm_scores, s=200, c="red", marker="x", linewidths=3, label="Drift alarm", zorder=5)
    plt.xlabel("Day")
    plt.ylabel("Drift Score")
    plt.title("Distribution Drift Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_attack_rate_by_day(
    df: pd.DataFrame,
    out_path: Path,
    *,
    domain_col: str = "day",
    label_col: str = "label",
) -> None:
    _ensure_dir(out_path.parent)
    daily = df.groupby(domain_col)[label_col].agg(["count", "sum"]).reset_index()
    daily.columns = [domain_col, "n_samples", "n_attacks"]
    daily["attack_rate"] = daily["n_attacks"] / daily["n_samples"]
    plt.figure(figsize=(10, 5))
    plt.plot(daily[domain_col], daily["attack_rate"], "o-", linewidth=2, markersize=5)
    plt.xlabel("Day")
    plt.ylabel("Attack Rate")
    plt.title("Attack Rate by Day")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_performance_vs_day(
    metrics_df: pd.DataFrame,
    out_path: Path,
    *,
    domain_col: str = "day",
    metric_col: str = "roc_auc",
    drift_alarms: pd.DataFrame | None = None,
) -> None:
    _ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 5))
    days = metrics_df[domain_col].unique()
    for model in metrics_df["model"].unique():
        model_data = metrics_df[metrics_df["model"] == model]
        values = [model_data[model_data[domain_col] == d][metric_col].values[0] if len(model_data[model_data[domain_col] == d]) > 0 else np.nan for d in days]
        plt.plot(days, values, "o-", label=model, linewidth=2, markersize=5)
    if drift_alarms is not None:
        alarm_days = drift_alarms[drift_alarms["alarm"] == 1][domain_col].values
        if len(alarm_days) > 0:
            plt.axvline(x=alarm_days[0], color="red", linestyle="--", alpha=0.5, label="Drift alarm")
            for ad in alarm_days[1:]:
                plt.axvline(x=ad, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Day")
    plt.ylabel(metric_col.replace("_", " ").title())
    plt.title(f"{metric_col.replace('_', ' ').title()} vs Day")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

