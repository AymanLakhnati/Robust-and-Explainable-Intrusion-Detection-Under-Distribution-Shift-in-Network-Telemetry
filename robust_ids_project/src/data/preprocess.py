from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    feature_names: list[str]


def infer_feature_columns(
    df: pd.DataFrame,
    *,
    label_col: str,
    domain_col: str,
    drop_cols: list[str] | None = None,
) -> list[str]:
    drop = set(drop_cols or [])
    drop |= {label_col, domain_col}
    cols = [c for c in df.columns if c not in drop]
    if len(cols) == 0:
        raise ValueError("No feature columns left after dropping label/domain/drop_cols.")
    return cols


def infer_categorical_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    categorical_max_unique: int = 50,
) -> tuple[list[str], list[str]]:
    cat_cols: list[str] = []
    num_cols: list[str] = []
    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            cat_cols.append(c)
            continue
        nunique = int(s.dropna().nunique())
        if nunique <= categorical_max_unique and nunique > 1 and not pd.api.types.is_float_dtype(s):
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return cat_cols, num_cols


def build_preprocessor(
    df_train: pd.DataFrame,
    *,
    feature_cols: list[str],
    categorical_max_unique: int = 50,
    scale_numeric: bool = True,
) -> PreprocessArtifacts:
    cat_cols, num_cols = infer_categorical_columns(
        df_train, feature_cols, categorical_max_unique=categorical_max_unique
    )

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric and len(num_cols) > 0:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if len(num_cols) > 0:
        transformers.append(("num", numeric_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.fit(df_train[feature_cols])

    feature_names = list(preprocessor.get_feature_names_out())
    return PreprocessArtifacts(preprocessor=preprocessor, feature_names=feature_names)


def transform(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
) -> np.ndarray:
    X = preprocessor.transform(df[feature_cols])
    X = np.asarray(X)
    return X

