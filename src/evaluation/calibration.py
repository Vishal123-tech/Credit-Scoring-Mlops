"""Probability calibration fitted only on out-of-fold predictions."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Platt-style sigmoid calibration for a binary default probability."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: LogisticRegression | None = None

    @staticmethod
    def _logit(probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped))

    def fit(self, y_true: np.ndarray, raw_probabilities: np.ndarray) -> "ProbabilityCalibrator":
        y_true = np.asarray(y_true)
        raw_probabilities = np.asarray(raw_probabilities, dtype=float)
        if len(y_true) != len(raw_probabilities):
            raise ValueError("y_true and raw_probabilities must have the same length")
        if len(np.unique(y_true)) < 2:
            raise ValueError("Calibration requires both default and non-default observations")

        self.model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=self.seed)
        self.model.fit(self._logit(raw_probabilities).reshape(-1, 1), y_true)
        return self

    def transform(self, raw_probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Calibrator is not fitted")
        logits = self._logit(raw_probabilities).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]


class CalibratedModel:
    """Serializable scoring wrapper that preserves the base model interface."""

    def __init__(self, base_model, calibrator: ProbabilityCalibrator):
        self.base_model = base_model
        # Preserve the historical artifact contract: `.model` is the underlying estimator.
        # The full base wrapper remains available as `.base_model` for feature alignment.
        self.model = base_model.model
        self.calibrator = calibrator
        self.features_ = base_model.features_

    def transform(self, X):
        return self.base_model.transform(X)

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base_model.predict_proba(X)
        return self.calibrator.transform(raw)
