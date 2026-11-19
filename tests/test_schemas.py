"""Unit tests for DataSentry Pydantic schemas."""

import json

from datasentry.schemas import (
    AnomalyReport,
    AxisType,
    DataSentryReport,
    Finding,
    QualityReport,
    Severity,
)


def test_finding_serialization():
    finding = Finding(
        finding_id="FND-001",
        axis=AxisType.QUALITY,
        severity=Severity.HIGH,
        column="age",
        metric_name="null_ratio",
        metric_value=0.25,
        threshold=0.10,
        description="High null percentage in age.",
    )
    data = finding.model_dump()
    assert data["finding_id"] == "FND-001"
    assert data["axis"] == "quality"
    assert data["severity"] == "high"

    # Test JSON round-trip
    json_str = finding.model_dump_json()
    reloaded = Finding.model_validate_json(json_str)
    assert reloaded.metric_value == 0.25


def test_datasentry_report_json_export():
    quality = QualityReport(
        passed=True,
        total_rows=100,
        total_columns=3,
        duplicate_rows=0,
        duplicate_percentage=0.0,
        schema_matched=True,
    )
    anomaly = AnomalyReport(
        total_evaluated=100,
        anomaly_count=2,
        anomaly_percentage=0.02,
        method="isolation_forest",
    )
    report = DataSentryReport(
        report_id="RPT-TEST-001",
        timestamp="2026-11-19T00:00:00Z",
        dataset_name="test_dataset",
        quality=quality,
        anomaly=anomaly,
        drift=None,
        overall_status="PASS",
    )
    raw_json = report.model_dump_json()
    parsed = json.loads(raw_json)
    assert parsed["report_id"] == "RPT-TEST-001"
    assert parsed["overall_status"] == "PASS"
