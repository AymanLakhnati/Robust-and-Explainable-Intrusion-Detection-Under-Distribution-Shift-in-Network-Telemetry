from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
from scipy import stats


DriftMethod = Literal["ks", "psi", "both"]


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    ks_stat: float | None
    ks_pvalue: float | None
    psi: float | None


def _compute_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    *,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.unique(np.percentile(ref, quantiles))
    if bins.size <= 1:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=bins)
    cur_counts, _ = np.histogram(cur, bins=bins)
    ref_prop = ref_counts / (ref_counts.sum() + eps)
    cur_prop = cur_counts / (cur_counts.sum() + eps)
    ref_prop = np.clip(ref_prop, eps, 1.0)
    cur_prop = np.clip(cur_prop, eps, 1.0)
    psi = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return float(psi)


def compute_drift(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    *,
    feature_cols: List[str],
    method: DriftMethod = "both",
    psi_bins: int = 10,
) -> Dict[str, FeatureDriftResult]:
    results: Dict[str, FeatureDriftResult] = {}
    for feat in feature_cols:
        if feat not in df_ref.columns or feat not in df_cur.columns:
            continue
        s_ref = df_ref[feat]
        s_cur = df_cur[feat]
        ks_stat = ks_p = None
        psi_val = None

        if method in ("ks", "both") and pd.api.types.is_numeric_dtype(s_ref) and pd.api.types.is_numeric_dtype(s_cur):
            try:
                ks_res = stats.ks_2samp(s_ref.dropna(), s_cur.dropna())
                ks_stat = float(ks_res.statistic)
                ks_p = float(ks_res.pvalue)
            except Exception:
                ks_stat = ks_p = None

        if method in ("psi", "both") and pd.api.types.is_numeric_dtype(s_ref) and pd.api.types.is_numeric_dtype(s_cur):
            try:
                psi_val = _compute_psi(s_ref.to_numpy(dtype=float), s_cur.to_numpy(dtype=float), n_bins=psi_bins)
            except Exception:
                psi_val = None

        results[feat] = FeatureDriftResult(feature=feat, ks_stat=ks_stat, ks_pvalue=ks_p, psi=psi_val)
    return results

