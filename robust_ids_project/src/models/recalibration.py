from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalibMethod = Literal["platt", "isotonic"]


@dataclass
class ProbabilityCalibrator:
    base_model: Any
    method: CalibMethod = "platt"
    _calibrator: Any | None = None

    def fit(
        self,
        X_calib: np.ndarray,
        y_calib: np.ndarray,
    ) -> "ProbabilityCalibrator":
        proba = self.base_model.predict_proba(X_calib)[:, 1]
        y_calib = np.asarray(y_calib).astype(int)

        if self.method == "platt":
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(proba.reshape(-1, 1), y_calib)
            self._calibrator = lr
        else:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(proba, y_calib)
            self._calibrator = ir
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_p = self.base_model.predict_proba(X)[:, 1]
        if self._calibrator is None:
            p1 = base_p
        else:
            if self.method == "platt":
                p1 = self._calibrator.predict_proba(base_p.reshape(-1, 1))[:, 1]
            else:
                p1 = self._calibrator.predict(base_p)
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

