"""Unit tests for FaultInjector (Phase 1)."""

import pytest

from datasentry.evaluation.datasets import load_adult_census
from datasentry.evaluation.fault_injection import FaultInjector, FaultType


@pytest.fixture
def clean_dataset():
    df, schema = load_adult_census(n_samples=200, random_state=42)
    return df, schema


def test_inject_schema_drop(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_schema_drop(df, "age")
    assert res.fault_type == FaultType.SCHEMA_DROP
    assert "age" not in res.corrupted_df.columns
    assert len(res.corrupted_df) == len(df)


def test_inject_schema_rename(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_schema_rename(df, "age", "client_age")
    assert res.fault_type == FaultType.SCHEMA_RENAME
    assert "age" not in res.corrupted_df.columns
    assert "client_age" in res.corrupted_df.columns


def test_inject_missingness_spike(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_missingness_spike(df, "hours_per_week", missing_rate=0.40)
    assert res.fault_type == FaultType.MISSINGNESS_SPIKE
    null_ratio = res.corrupted_df["hours_per_week"].isna().mean()
    assert 0.35 <= null_ratio <= 0.45


def test_inject_unit_error(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    orig_mean = df["hours_per_week"].mean()
    res = injector.inject_unit_error(df, "hours_per_week", conversion="celsius_to_fahrenheit")
    assert res.fault_type == FaultType.UNIT_ERROR
    new_mean = res.corrupted_df["hours_per_week"].mean()
    assert abs(new_mean - (orig_mean * 1.8 + 32.0)) < 1e-4


def test_inject_out_of_range(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_out_of_range(df, "hours_per_week", fraction=0.10)
    assert res.fault_type == FaultType.OUT_OF_RANGE
    assert len(res.target_rows) > 0
    # Check that injected values are outside original range
    injected_vals = res.corrupted_df.loc[res.target_rows, "hours_per_week"]
    assert (injected_vals > df["hours_per_week"].max()).any() or (injected_vals < df["hours_per_week"].min()).any()


def test_inject_duplicate_rows(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_duplicate_rows(df, fraction=0.10)
    assert res.fault_type == FaultType.DUPLICATE_ROWS
    assert len(res.corrupted_df) > len(df)
    assert res.corrupted_df.duplicated().sum() >= int(len(df) * 0.10)


def test_inject_covariate_shift(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_covariate_shift(df, "hours_per_week", shift_std=3.0)
    assert res.fault_type == FaultType.COVARIATE_SHIFT
    orig_mean = df["hours_per_week"].mean()
    new_mean = res.corrupted_df["hours_per_week"].mean()
    assert new_mean > orig_mean + 2.5 * df["hours_per_week"].std()


def test_inject_correlated_feature_breakage(clean_dataset):
    df, _ = clean_dataset
    injector = FaultInjector()
    res = injector.inject_correlated_feature_breakage(df, "education_num", "hours_per_week")
    assert res.fault_type == FaultType.CORRELATED_FEATURE_BREAKAGE
    assert len(res.corrupted_df) == len(df)
