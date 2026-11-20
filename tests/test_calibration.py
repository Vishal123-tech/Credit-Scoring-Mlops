import numpy as np
import pytest

from src.evaluation.calibration import ProbabilityCalibrator


def test_sigmoid_calibrator_is_fitted_and_monotonic():
    y = np.array([0, 0, 0, 1, 1, 1])
    raw = np.array([0.05, 0.15, 0.30, 0.55, 0.75, 0.95])
    calibrated = ProbabilityCalibrator(seed=42).fit(y, raw).transform(raw)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert np.all(np.diff(calibrated) >= 0)


def test_calibrator_rejects_single_class():
    with pytest.raises(ValueError, match="both default"):
        ProbabilityCalibrator().fit(np.zeros(4), np.full(4, 0.2))
