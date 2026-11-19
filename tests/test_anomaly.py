"""Unit tests for AnomalyDetector (Axis 2)."""

import numpy as np
import pandas as pd
import pytest

from datasentry.core.anomaly import AnomalyDetector


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    # 100 normal points
    normal_x = np.random.normal(loc=50.0, scale=5.0, size=100)
    normal_y = np.random.normal(loc=100.0, scale=10.0, size=100)

    # 5 extreme outliers
    outlier_x = np.array([200.0, 250.0, -100.0, 300.0, 500.0])
    outlier_y = np.array([500.0, 600.0, -200.0, 700.0, 1000.0])

    df = pd.DataFrame({
        "feat_a": np.concatenate([normal_x, outlier_x]),
        "feat_b": np.concatenate([normal_y, outlier_y]),
    })
    return df


def test_isolation_forest_detects_outliers(synthetic_data):
    detector = AnomalyDetector(method="isolation_forest", contamination=0.05, random_state=42)
    report = detector.detect(synthetic_data)
    assert report.total_evaluated == 105
    assert report.anomaly_count >= 5

    # Check that extreme rows (100 to 104) are identified
    flagged_indices = {r.row_index for r in report.anomalies if r.is_anomaly}
    assert any(idx in flagged_indices for idx in [100, 101, 102, 103, 104])


def test_iqr_detector(synthetic_data):
    detector = AnomalyDetector(method="iqr")
    report = detector.detect(synthetic_data)
    assert report.method == "iqr"
    flagged_indices = {r.row_index for r in report.anomalies if r.is_anomaly}
    for idx in [100, 101, 102, 103, 104]:
        assert idx in flagged_indices


def test_mahalanobis_detector(synthetic_data):
    detector = AnomalyDetector(method="mahalanobis", contamination=0.05)
    report = detector.detect(synthetic_data)
    assert report.method == "mahalanobis"
    flagged_indices = {r.row_index for r in report.anomalies if r.is_anomaly}
    for idx in [100, 101, 102, 103, 104]:
        assert idx in flagged_indices


def test_anomaly_contributing_features(synthetic_data):
    detector = AnomalyDetector(method="isolation_forest", contamination=0.05)
    report = detector.detect(synthetic_data)
    flagged = [r for r in report.anomalies if r.is_anomaly]
    assert len(flagged) > 0
    # Contributing features dictionary should not be empty
    assert len(flagged[0].contributing_features) > 0
    assert "feat_a" in flagged[0].contributing_features or "feat_b" in flagged[0].contributing_features
