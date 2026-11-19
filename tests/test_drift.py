"""Unit tests for DriftDetector (Axis 3)."""

import numpy as np
import pandas as pd

from datasentry.core.drift import DriftDetector, calculate_psi


def test_psi_identical_distributions():
    np.random.seed(42)
    a = np.random.normal(loc=0.0, scale=1.0, size=500)
    b = np.random.normal(loc=0.0, scale=1.0, size=500)
    psi = calculate_psi(a, b)
    assert psi < 0.10  # PSI < 0.1 indicates no shift


def test_psi_shifted_distributions():
    np.random.seed(42)
    a = np.random.normal(loc=0.0, scale=1.0, size=500)
    b = np.random.normal(loc=2.5, scale=1.0, size=500)  # Significant shift
    psi = calculate_psi(a, b)
    assert psi > 0.25  # PSI > 0.25 indicates significant shift


def test_drift_detector_numeric_shift():
    np.random.seed(42)
    ref = pd.DataFrame({
        "temperature": np.random.normal(20.0, 2.0, size=200),
        "pressure": np.random.normal(1013.0, 5.0, size=200),
        "humidity": np.random.normal(50.0, 5.0, size=200),
    })
    # Current batch has drifted temperature
    cur = pd.DataFrame({
        "temperature": np.random.normal(35.0, 2.0, size=200),  # Drifted
        "pressure": np.random.normal(1013.0, 5.0, size=200),  # Stable
        "humidity": np.random.normal(50.0, 5.0, size=200),    # Stable
    })

    detector = DriftDetector(p_val_threshold=0.05, dataset_drift_threshold=0.50)
    report = detector.detect(ref, cur)
    assert report.total_features == 3
    assert report.drifted_features_count == 1
    assert "temperature" in report.metrics
    assert report.metrics["temperature"].drifted is True
    assert report.metrics["pressure"].drifted is False
    assert report.dataset_drift is False  # 1/3 = 33% < 50% threshold


def test_drift_detector_dataset_level_shift():
    np.random.seed(42)
    ref = pd.DataFrame({
        "feat_1": np.random.normal(0, 1, 200),
        "feat_2": np.random.normal(10, 2, 200),
    })
    # Both features shifted
    cur = pd.DataFrame({
        "feat_1": np.random.normal(5, 1, 200),
        "feat_2": np.random.normal(25, 2, 200),
    })
    detector = DriftDetector(dataset_drift_threshold=0.50)
    report = detector.detect(ref, cur)
    assert report.drifted_features_count == 2
    assert report.dataset_drift is True
    assert any(f.finding_id == "FND-DRFT-DATASET" for f in report.findings)


def test_drift_detector_categorical():
    ref = pd.DataFrame({"city": ["NY"] * 100 + ["LA"] * 100})
    cur = pd.DataFrame({"city": ["NY"] * 10 + ["LA"] * 190})  # Heavy shift toward LA

    detector = DriftDetector(p_val_threshold=0.05)
    report = detector.detect(ref, cur)
    assert report.metrics["city"].drifted is True
