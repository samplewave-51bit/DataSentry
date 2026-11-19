"""Unit tests for PyTorch Tabular Autoencoder (Phase 3)."""

import numpy as np
import pandas as pd
import pytest

from datasentry.models.autoencoder import AutoencoderAnomalyDetector, TabularAutoencoder


@pytest.fixture
def sample_numeric_df():
    np.random.seed(42)
    # 150 normal points
    normal = np.random.normal(loc=10.0, scale=2.0, size=(150, 4))
    # 10 extreme outliers
    outliers = np.random.normal(loc=50.0, scale=5.0, size=(10, 4))
    all_data = np.vstack([normal, outliers])

    df = pd.DataFrame(all_data, columns=["f1", "f2", "f3", "f4"])
    return df


def test_tabular_autoencoder_architecture():
    model = TabularAutoencoder(input_dim=5, latent_dim=2, hidden_dim=16)
    import torch
    x = torch.randn(8, 5)
    out = model(x)
    assert out.shape == (8, 5)


def test_autoencoder_detector_fit_and_detect(sample_numeric_df):
    detector = AutoencoderAnomalyDetector(epochs=15, contamination=0.05, random_state=42)
    # Fit on first 100 normal points
    detector.fit(sample_numeric_df.iloc[:100])

    # Detect on all points (including outliers at indices 150-159)
    report = detector.detect(sample_numeric_df)

    assert report.total_evaluated == 160
    assert report.anomaly_count >= 8
    assert report.method == "pytorch_autoencoder"

    # Verify outlier rows are flagged
    flagged_indices = {r.row_index for r in report.anomalies if r.is_anomaly}
    assert any(idx in flagged_indices for idx in range(150, 160))

    # Verify explainable feature contributions
    flagged_records = [r for r in report.anomalies if r.is_anomaly]
    assert len(flagged_records[0].contributing_features) > 0


def test_seed_reproducibility(sample_numeric_df):
    det1 = AutoencoderAnomalyDetector(epochs=10, random_state=42).fit(sample_numeric_df.iloc[:50])
    score1 = det1.detect(sample_numeric_df.iloc[50:60]).anomalies[0].anomaly_score

    det2 = AutoencoderAnomalyDetector(epochs=10, random_state=42).fit(sample_numeric_df.iloc[:50])
    score2 = det2.detect(sample_numeric_df.iloc[50:60]).anomalies[0].anomaly_score

    assert abs(score1 - score2) < 1e-4
