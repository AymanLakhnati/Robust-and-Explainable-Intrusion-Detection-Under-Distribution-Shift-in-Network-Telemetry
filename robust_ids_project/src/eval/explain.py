from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance

from .plots import plot_feature_importance


def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    res = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    importances = res.importances_mean
    return {feat: float(imp) for feat, imp in zip(feature_names, importances)}


def rank_correlation_topk(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    *,
    top_k: int = 20,
) -> Tuple[float, float]:
    features = list(scores_a.keys())
    if len(features) < top_k:
        top_k = len(features)
    vals_a = np.array([scores_a.get(f, 0.0) for f in features])
    vals_b = np.array([scores_b.get(f, 0.0) for f in features])
    idx = np.argsort(vals_a)[::-1][:top_k]
    a = vals_a[idx]
    b = vals_b[idx]
    if a.size <= 1:
        return 1.0, 1.0
    spearman, _ = stats.spearmanr(a, b)
    kendall, _ = stats.kendalltau(a, b)
    return float(spearman), float(kendall)


def explain_with_permutation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    *,
    random_state: int = 42,
    title: str = "Permutation Importance",
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = compute_permutation_importance(
        model,
        X,
        y,
        feature_names,
        random_state=random_state,
    )
    txt_path = out_dir / "permutation_importance.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for feat, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
            f.write(f"{feat}\t{score:.6f}\n")

    importances = np.array([scores[f] for f in feature_names], dtype=float)
    fig_path = out_dir / "permutation_importance.png"
    plot_feature_importance(feature_names, importances, fig_path, title=title)
    return scores


def explain_with_shap_tree(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    *,
    max_samples: int = 2000,
    title: str = "SHAP Summary",
) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    X_sample = X
    if X.shape[0] > max_samples:
        np.random.seed(42)
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X[idx]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        return

    shap_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values

    fig_path = out_dir / "shap_summary_bar.png"
    plt.figure(figsize=(10, max(6, 0.3 * len(feature_names))))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_local_cases(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: List[str],
    indices: List[int] | None,
    out_dir: Path,
    *,
    case_types: List[str] | None = None,
    top_k: int = 10,
) -> None:
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        tps = np.where((y_true == 1) & (y_pred == 1))[0]
        fps = np.where((y_true == 0) & (y_pred == 1))[0]
        indices = []
        case_types = []
        if len(tps) > 0:
            indices.append(int(tps[0]))
            case_types.append("TP")
        if len(fps) > 0:
            indices.append(int(fps[0]))
            case_types.append("FP")

    if len(indices) == 0:
        return

    explainer = shap.TreeExplainer(model)

    for idx, case_type in zip(indices, case_types or ["Case"] * len(indices)):
        x_sample = X[idx : idx + 1]
        try:
            shap_values = explainer.shap_values(x_sample)
            shap_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
        except Exception:
            continue

        top_idx = np.argsort(np.abs(shap_vals[0]))[::-1][:top_k]
        top_features = [feature_names[i] for i in top_idx]
        top_values = shap_vals[0][top_idx]

        plt.figure(figsize=(8, max(4, 0.3 * len(top_features))))
        y_pos = np.arange(len(top_features))
        colors = ["red" if v > 0 else "blue" for v in top_values]
        plt.barh(y_pos, top_values[::-1], color=colors[::-1])
        plt.yticks(y_pos, top_features[::-1])
        plt.xlabel("SHAP Value")
        plt.title(f"Local Explanation: {case_type} (Sample {idx})")
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
        plt.grid(alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(out_dir / f"shap_local_{case_type.lower()}_{idx}.png", dpi=150)
        plt.close()

