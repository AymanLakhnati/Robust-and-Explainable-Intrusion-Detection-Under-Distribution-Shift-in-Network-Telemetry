from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def make_logistic_regression(params: Dict[str, Any], *, random_seed: int) -> LogisticRegression:
    defaults: Dict[str, Any] = dict(
        C=1.0,
        max_iter=2000,
        class_weight=None,
        solver="liblinear",
    )
    defaults.update(params or {})
    defaults.setdefault("random_state", random_seed)
    return LogisticRegression(**defaults)


def make_random_forest(params: Dict[str, Any], *, random_seed: int) -> RandomForestClassifier:
    defaults: Dict[str, Any] = dict(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        n_jobs=-1,
        random_state=random_seed,
    )
    defaults.update(params or {})
    defaults.setdefault("random_state", random_seed)
    return RandomForestClassifier(**defaults)

