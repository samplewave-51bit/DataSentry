"""Unit tests for QualityValidator (Axis 1)."""

import pandas as pd
import pytest

from datasentry.core.quality import QualityValidator
from datasentry.schemas import Severity


@pytest.fixture
def clean_df():
    return pd.DataFrame({
        "age": [25, 30, 45, 22, 58, 33, 41],
        "income": [50000.0, 60000.0, 95000.0, 32000.0, 110000.0, 62000.0, 80000.0],
        "department": ["Sales", "Engineering", "Marketing", "Sales", "Engineering", "Sales", "Marketing"],
    })


def test_clean_df_passes(clean_df):
    validator = QualityValidator(
        schema={"age": "numeric", "income": "numeric", "department": "string"},
        constraints={"age": {"min": 18, "max": 75}},
    )
    report = validator.validate(clean_df)
    assert report.passed is True
    assert report.total_rows == 7
    assert report.duplicate_rows == 0
    assert len(report.findings) == 0


def test_detects_duplicates(clean_df):
    dup_df = pd.concat([clean_df, clean_df.iloc[[0]], clean_df.iloc[[0]]], ignore_index=True)
    validator = QualityValidator(duplicate_threshold=0.05)
    report = validator.validate(dup_df)
    assert report.duplicate_rows == 2
    assert report.duplicate_percentage > 0.05
    assert any(f.metric_name == "duplicate_ratio" for f in report.findings)


def test_detects_missing_mandatory_column(clean_df):
    dropped_df = clean_df.drop(columns=["income"])
    validator = QualityValidator(schema={"age": "numeric", "income": "numeric", "department": "string"})
    report = validator.validate(dropped_df)
    assert report.passed is False
    assert report.schema_matched is False
    assert any(f.severity == Severity.CRITICAL and f.column == "income" for f in report.findings)


def test_detects_high_null_rate(clean_df):
    corrupt_df = clean_df.copy()
    corrupt_df.loc[0:4, "age"] = None
    validator = QualityValidator(null_threshold=0.20)
    report = validator.validate(corrupt_df)
    assert any(f.column == "age" and f.metric_name == "null_ratio" for f in report.findings)


def test_detects_out_of_bounds_constraints(clean_df):
    violating_df = clean_df.copy()
    violating_df.loc[0, "age"] = -5  # Negative age
    violating_df.loc[1, "age"] = 150  # Over max
    validator = QualityValidator(constraints={"age": {"min": 0, "max": 100}})
    report = validator.validate(violating_df)
    col_metric = report.column_metrics["age"]
    assert col_metric.out_of_bounds_count == 2
    assert any(f.column == "age" and f.metric_name == "constraint_violations" for f in report.findings)


def test_empty_dataframe():
    validator = QualityValidator()
    report = validator.validate(pd.DataFrame())
    assert report.passed is False
    assert report.total_rows == 0
