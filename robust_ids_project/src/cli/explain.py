from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import transform
from src.eval.explain import (
    explain_local_cases,
    explain_with_permutation,
    explain_with_shap_tree,
    rank_correlation_topk,
)
from src.utils.config import deep_get, ensure_dir, load_yaml
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain models (permutation importance, optional SHAP).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    random_seed = int(deep_get(cfg, ["project", "random_seed"], 42))
    set_global_seed(random_seed)

    paths_cfg: Dict[str, Any] = deep_get(cfg, ["paths"], {})
    explain_cfg: Dict[str, Any] = deep_get(cfg, ["explain"], {})

    if not explain_cfg.get("enabled", True):
        return

    artifacts_dir = ensure_dir(paths_cfg.get("artifacts_dir", "artifacts"))
    results_dir = ensure_dir(paths_cfg.get("results_dir", "results"))
    explain_dir = ensure_dir(results_dir / "explain")

    test_df = pd.read_csv(artifacts_dir / "test.csv")
    meta = load_yaml(artifacts_dir / "preprocess_meta.yaml")
    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    label_col = meta["label_col"]
    domain_col = meta["domain_col"]
    feature_cols: List[str] = meta["feature_cols"]

    models_dir = artifacts_dir / "models"
    try:
        rf = joblib.load(models_dir / "rf_calibrated.joblib")
        base_rf = getattr(rf, "base_model", rf)
    except Exception:
        rf = joblib.load(models_dir / "rf_base.joblib")
        base_rf = rf

    X_test = transform(preprocessor, test_df, feature_cols=feature_cols)
    y_test = test_df[label_col].to_numpy(dtype=int)

    explain_with_permutation(
        model=rf,
        X=X_test,
        y=y_test,
        feature_names=feature_cols,
        out_dir=explain_dir,
        random_state=random_seed,
    )

    shap_enabled = explain_cfg.get("shap_enabled", True)
    if shap_enabled:
        explain_with_shap_tree(
            model=base_rf,
            X=X_test,
            feature_names=feature_cols,
            out_dir=explain_dir,
            title="SHAP Summary (Test Set)",
        )

        stable_day = explain_cfg.get("stable_day")
        drift_day = explain_cfg.get("drift_day")
        unique_domains = sorted(test_df[domain_col].unique())

        if stable_day is not None and stable_day in unique_domains:
            stable_mask = test_df[domain_col] == stable_day
            X_stable = transform(preprocessor, test_df[stable_mask], feature_cols=feature_cols)
            explain_with_shap_tree(
                model=base_rf,
                X=X_stable,
                feature_names=feature_cols,
                out_dir=explain_dir / "stable_day",
                title=f"SHAP Summary: Stable Day {stable_day}",
            )
            stable_importance = explain_with_permutation(
                model=rf,
                X=X_stable,
                y=test_df[stable_mask][label_col].to_numpy(dtype=int),
                feature_names=feature_cols,
                out_dir=explain_dir / "stable_day",
                random_state=random_seed,
                title=f"Permutation Importance: Stable Day {stable_day}",
            )

        if drift_day is not None and drift_day in unique_domains:
            drift_mask = test_df[domain_col] == drift_day
            X_drift = transform(preprocessor, test_df[drift_mask], feature_cols=feature_cols)
            explain_with_shap_tree(
                model=base_rf,
                X=X_drift,
                feature_names=feature_cols,
                out_dir=explain_dir / "drift_day",
                title=f"SHAP Summary: Drift Day {drift_day}",
            )
            drift_importance = explain_with_permutation(
                model=rf,
                X=X_drift,
                y=test_df[drift_mask][label_col].to_numpy(dtype=int),
                feature_names=feature_cols,
                out_dir=explain_dir / "drift_day",
                random_state=random_seed,
                title=f"Permutation Importance: Drift Day {drift_day}",
            )

            if stable_day is not None and stable_day in unique_domains and "stable_importance" in locals():
                top_k = int(explain_cfg.get("top_k_features", 20))
                spearman, kendall = rank_correlation_topk(stable_importance, drift_importance, top_k=top_k)
                stability_df = pd.DataFrame(
                    [
                        {
                            "stable_day": stable_day,
                            "drift_day": drift_day,
                            "spearman_topk": spearman,
                            "kendall_topk": kendall,
                        }
                    ]
                )
                stability_df.to_csv(explain_dir / "explanation_stability.csv", index=False)

    y_pred_test = rf.predict(X_test)
    explain_local_cases(
        model=base_rf,
        X=X_test,
        y_true=y_test,
        y_pred=y_pred_test,
        feature_names=feature_cols,
        indices=None,
        out_dir=explain_dir / "local_cases",
        top_k=int(explain_cfg.get("top_k_features", 20)),
    )

    unique_domains = sorted(test_df[domain_col].unique())
    top_k = int(explain_cfg.get("top_k_features", 20))
    stability_rows: List[Dict[str, Any]] = []

    for i in range(len(unique_domains) - 1):
        day_a = unique_domains[i]
        day_b = unique_domains[i + 1]
        mask_a = test_df[domain_col] == day_a
        mask_b = test_df[domain_col] == day_b

        if mask_a.sum() == 0 or mask_b.sum() == 0:
            continue

        X_a = transform(preprocessor, test_df[mask_a], feature_cols=feature_cols)
        y_a = test_df[mask_a][label_col].to_numpy(dtype=int)
        X_b = transform(preprocessor, test_df[mask_b], feature_cols=feature_cols)
        y_b = test_df[mask_b][label_col].to_numpy(dtype=int)

        imp_a = explain_with_permutation(rf, X_a, y_a, feature_cols, random_state=random_seed)
        imp_b = explain_with_permutation(rf, X_b, y_b, feature_cols, random_state=random_seed)

        spearman, kendall = rank_correlation_topk(imp_a, imp_b, top_k=top_k)
        stability_rows.append(
            {
                "day_a": day_a,
                "day_b": day_b,
                "spearman_topk": spearman,
                "kendall_topk": kendall,
            }
        )

    if stability_rows:
        stability_df = pd.DataFrame(stability_rows)
        stability_df.to_csv(explain_dir / "explanation_stability_consecutive_days.csv", index=False)


if __name__ == "__main__":
    main()

