from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p}")
    return df


def coerce_binary_label(
    s: pd.Series,
    *,
    positive_label: str | int | float | bool = 1,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)

    if pd.api.types.is_numeric_dtype(s):
        uniq = pd.Series(s.dropna().unique()).sort_values()
        if len(uniq) == 2:
            if positive_label in set(uniq.tolist()):
                return (s == positive_label).astype(int)
            return (s == uniq.iloc[-1]).astype(int)
        if set(uniq.tolist()).issubset({0, 1}):
            return s.fillna(0).astype(int)
        raise ValueError("Numeric label column must be binary (two unique values).")

    s2 = s.astype(str).str.strip().str.lower()
    pos = str(positive_label).strip().lower()
    return (s2 == pos).astype(int)


def add_day_from_timestamp(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    domain_col: str = "day",
) -> pd.DataFrame:
    if domain_col in df.columns:
        return df
    if timestamp_col not in df.columns:
        raise KeyError(f"timestamp_col '{timestamp_col}' not found in dataframe")
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.copy()
    df[domain_col] = ts.dt.date.astype(str)
    return df

