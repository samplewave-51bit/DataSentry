"""Unit tests for TimeSeriesDetector (Phase 2)."""

import numpy as np
import pytest

from datasentry.detectors.timeseries import SeasonalResidualForecaster, TimeSeriesDetector


@pytest.fixture
def synthetic_series():
    np.random.seed(42)
    # 24-step daily seasonality over 10 days = 240 steps
    n_steps = 240
    t = np.arange(n_steps)
    trend = 0.05 * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 1.0, n_steps)
    y = 50.0 + trend + seasonal + noise
    return y


def test_seasonal_residual_forecaster_fit_and_interval(synthetic_series):
    forecaster = SeasonalResidualForecaster(period=24, alpha=0.05)
    forecaster.fit(synthetic_series)

    # In-sample coverage should be close to 95%
    assert 0.90 <= forecaster.calibrated_coverage <= 0.98

    # Predict next 24 steps
    preds, lower, upper = forecaster.predict_interval(start_idx=len(synthetic_series), horizon=24)
    assert len(preds) == 24
    assert (upper > lower).all()


def test_timeseries_detector_clean_window(synthetic_series):
    ref_y = synthetic_series[:168]  # First 7 days
    cur_y = synthetic_series[168:]  # Last 3 days (nominal continuation)

    detector = TimeSeriesDetector(period=24, interval_alpha=0.05)
    report = detector.detect(current_data=cur_y, reference_data=ref_y)

    assert report.total_points == len(cur_y)
    assert report.is_drift is False
    # Empirical coverage on nominal continuation should remain healthy (>= 80%)
    assert report.empirical_coverage >= 0.80
    assert report.forecaster_model == "seasonal_residual_forecaster"
    assert report.baseline_coverage > 0.0


def test_timeseries_detector_catches_anomalies_and_drift(synthetic_series):
    ref_y = synthetic_series[:168]
    cur_y = synthetic_series[168:].copy()

    # Inject severe upward shift and localized spike anomalies
    cur_y += 25.0  # Distribution drift
    cur_y[10] += 100.0  # Extreme anomaly point

    detector = TimeSeriesDetector(period=24, interval_alpha=0.05, drift_coverage_threshold=0.85)
    report = detector.detect(current_data=cur_y, reference_data=ref_y)

    assert report.is_drift is True
    assert report.anomaly_count > 0
    assert report.empirical_coverage < 0.50
    assert any(f.finding_id == "FND-TS-DRIFT-001" for f in report.findings)
