"""Unit tests for PyTorch Domain Classifier Drift Scorer (Phase 3)."""

import numpy as np
import pandas as pd
import pytest

from datasentry.models.domain_classifier import DomainClassifierDriftScorer, TabularDomainClassifier


@pytest.fixture
def clean_and_shifted_batches():
    np.random.seed(42)
    # Reference: 100 points
    ref_vals = np.random.normal(loc=20.0, scale=3.0, size=(100, 3))
    ref_df = pd.DataFrame(ref_vals, columns=["feat_x", "feat_y", "feat_z"])

    # Clean continuation: same distribution
    cur_clean_vals = np.random.normal(loc=20.0, scale=3.0, size=(100, 3))
    cur_clean_df = pd.DataFrame(cur_clean_vals, columns=["feat_x", "feat_y", "feat_z"])

    # Shifted: significantly shifted mean
    cur_shifted_vals = np.random.normal(loc=35.0, scale=3.0, size=(100, 3))
    cur_shifted_df = pd.DataFrame(cur_shifted_vals, columns=["feat_x", "feat_y", "feat_z"])

    return ref_df, cur_clean_df, cur_shifted_df


def test_domain_classifier_architecture():
    import torch
    model = TabularDomainClassifier(input_dim=4, hidden_dim=16)
    x = torch.randn(10, 4)
    logits = model(x)
    assert logits.shape == (10, 1)


def test_domain_classifier_drift_scoring(clean_and_shifted_batches):
    ref_df, clean_df, shifted_df = clean_and_shifted_batches
    scorer = DomainClassifierDriftScorer(epochs=15, auc_threshold=0.65, random_state=42)

    # 1. Clean batches should yield AUC close to 0.50 (no drift)
    auc_clean, is_drift_clean, metric_clean, _ = scorer.score(ref_df, clean_df)
    assert auc_clean < 0.65
    assert is_drift_clean is False
    assert metric_clean.drifted is False

    # 2. Shifted batches should yield high AUC (> 0.85)
    auc_shifted, is_drift_shifted, metric_shifted, findings = scorer.score(ref_df, shifted_df)
    assert auc_shifted >= 0.85
    assert is_drift_shifted is True
    assert metric_shifted.drifted is True
    assert len(findings) > 0
    assert any(f.finding_id == "FND-DRFT-DL-001" for f in findings)
