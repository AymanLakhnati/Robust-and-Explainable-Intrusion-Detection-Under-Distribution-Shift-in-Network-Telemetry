from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class ConformalAbstainer:
    base_model: Any
    alpha: float = 0.1
    abstain_if_ambiguous: bool = True
    _nonconformity_scores: np.ndarray | None = None

    def fit(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "ConformalAbstainer":
        proba = self.base_model.predict_proba(X_calib)
        y_calib = np.asarray(y_calib).astype(int)
        p_true = proba[np.arange(len(y_calib)), y_calib]
        self._nonconformity_scores = 1.0 - p_true
        return self

    def _predict_set_single(self, p: np.ndarray) -> List[int]:
        if self._nonconformity_scores is None:
            return [int(np.argmax(p))]

        alphas = self._nonconformity_scores
        n = len(alphas)
        q = np.quantile(alphas, 1 - self.alpha, interpolation="higher")

        pred_set: List[int] = []
        for y in (0, 1):
            alpha_test = 1.0 - p[y]
            p_value = (np.sum(alphas >= alpha_test) + 1.0) / (n + 1.0)
            if p_value > self.alpha:
                pred_set.append(y)
        if not pred_set:
            pred_set = [int(np.argmax(p))]
        return pred_set

    def predict_sets(self, X: np.ndarray) -> list[list[int]]:
        proba = self.base_model.predict_proba(X)
        return [self._predict_set_single(p) for p in proba]

    def predict(self, X: np.ndarray) -> np.ndarray:
        sets = self.predict_sets(X)
        preds = []
        for s in sets:
            if len(s) == 1:
                preds.append(s[0])
            else:
                if self.abstain_if_ambiguous:
                    preds.append(-1)
                else:
                    preds.append(max(s))
        return np.asarray(preds, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)

