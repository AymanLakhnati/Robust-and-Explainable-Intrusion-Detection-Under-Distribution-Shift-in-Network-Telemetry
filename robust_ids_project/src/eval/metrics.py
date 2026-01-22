from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float = 0.01,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    mask = fpr <= target_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def pauc_low_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    max_fpr: float = 0.01,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    if not np.any(mask):
        return 0.0
    pauc = auc(fpr[mask], tpr[mask])
    return float(pauc / max_fpr)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    prob_true, prob_pred = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if len(prob_pred) == 0:
        return 0.0
    return float(np.mean(np.abs(prob_true - prob_pred)))


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    prob_true = []
    prob_pred = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            prob_true.append(accuracy_in_bin)
            prob_pred.append(avg_confidence_in_bin)
    return np.array(prob_pred), np.array(prob_true)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float = 0.01,
    low_fpr_targets: List[float] | None = None,
    low_fpr_pauc_max: float | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    metrics: Dict[str, float] = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    metrics["pr_auc"] = float(auc(recall, precision))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics[f"recall_at_fpr_{target_fpr:.3f}"] = float(
        recall_at_fixed_fpr(y_true, y_score, target_fpr=target_fpr)
    )

    if low_fpr_targets:
        for t in low_fpr_targets:
            metrics[f"tpr_at_fpr_{t:.0e}"] = float(
                recall_at_fixed_fpr(y_true, y_score, target_fpr=t)
            )

    if low_fpr_pauc_max:
        metrics["pauc_low_fpr"] = float(
            pauc_low_fpr(y_true, y_score, max_fpr=low_fpr_pauc_max)
        )

    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> Dict[str, float]:
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob, n_bins=n_bins))
    return {"brier_score": brier, "ece": ece}

