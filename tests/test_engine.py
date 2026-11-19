"""Unit tests for DataSentryEngine orchestrator."""

import numpy as np
import pandas as pd

from datasentry.core.engine import DataSentryEngine


def test_engine_clean_data_passes():
    np.random.seed(42)
    df = pd.DataFrame({
        "feature_1": np.random.normal(10.0, 1.0, 100),
        "feature_2": np.random.normal(50.0, 5.0, 100),
    })
    engine = DataSentryEngine(
        schema={"feature_1": "numeric", "feature_2": "numeric"},
        contamination=0.01,
    )
    report = engine.scan(current_df=df, reference_df=df)
    assert report.quality.passed is True
    assert report.drift is not None
    assert report.drift.dataset_drift is False
    assert report.overall_status in ("PASS", "WARN")


def test_engine_corrupt_data_fails():
    ref_df = pd.DataFrame({
        "feature_1": np.random.normal(10.0, 1.0, 100),
        "feature_2": np.random.normal(50.0, 5.0, 100),
    })
    # Batch missing mandatory column
    bad_df = pd.DataFrame({
        "feature_1": np.random.normal(10.0, 1.0, 100),
    })
    engine = DataSentryEngine(
        schema={"feature_1": "numeric", "feature_2": "numeric"}
    )
    report = engine.scan(current_df=bad_df, reference_df=ref_df)
    assert report.overall_status == "FAIL"
    assert report.quality.passed is False
