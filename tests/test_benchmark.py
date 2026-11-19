"""Unit tests for BenchmarkRunner and Baselines (Phase 1)."""

from datasentry.evaluation.baselines import (
    DataSentryEngineBaseline,
    EvidentlyBaseline,
    GreatExpectationsBaseline,
    NaiveKSBaseline,
)
from datasentry.evaluation.datasets import load_sensor_telemetry


def test_baselines_execution():
    df, schema = load_sensor_telemetry(n_samples=100, random_state=42)
    ref_df = df.iloc[:50]
    cur_df = df.iloc[50:]

    detectors = [
        DataSentryEngineBaseline(),
        EvidentlyBaseline(),
        GreatExpectationsBaseline(),
        NaiveKSBaseline(),
    ]

    for det in detectors:
        res = det.run(current_df=cur_df, reference_df=ref_df, schema=schema)
        assert "detector" in res
        assert "detected_fault" in res
        assert "latency_ms" in res
        assert res["latency_ms"] >= 0.0
