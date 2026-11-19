"""
DataSentry Core - Engine Orchestrator

Consolidates Data-Quality Validation, Record-Level Anomaly Detection,
and Dataset Drift Detection into a single structured DataSentryReport.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from datasentry.core.anomaly import AnomalyDetector
from datasentry.core.drift import DriftDetector
from datasentry.core.quality import QualityValidator
from datasentry.detectors.text import TextDriftDetector
from datasentry.detectors.timeseries import TimeSeriesDetector
from datasentry.schemas import (
    DataSentryReport,
    Finding,
    Severity,
    TextDriftReport,
    TimeSeriesReport,
)


class DataSentryEngine:
    """Primary monitoring and diagnostic engine for tabular data."""

    def __init__(
        self,
        schema: Optional[Dict[str, str]] = None,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        anomaly_method: str = "isolation_forest",
        contamination: float = 0.05,
        drift_p_val_threshold: float = 0.05,
        drift_psi_threshold: float = 0.20,
    ):
        self.quality_validator = QualityValidator(
            schema=schema,
            constraints=constraints,
        )
        self.anomaly_detector = AnomalyDetector(
            method=anomaly_method,
            contamination=contamination,
        )
        self.drift_detector = DriftDetector(
            p_val_threshold=drift_p_val_threshold,
            psi_threshold=drift_psi_threshold,
        )

    def scan(
        self,
        current_df: Optional[pd.DataFrame] = None,
        reference_df: Optional[pd.DataFrame] = None,
        dataset_name: str = "dataset_batch",
        timeseries_current: Optional[Any] = None,
        timeseries_reference: Optional[Any] = None,
        text_current: Optional[Any] = None,
        text_reference: Optional[Any] = None,
    ) -> DataSentryReport:
        """Run all active diagnostic axes and compile report."""
        report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Quality Validation & Anomaly Detection (Tabular)
        df_for_quality = current_df if current_df is not None else pd.DataFrame()
        quality_report = self.quality_validator.validate(df_for_quality)
        anomaly_report = self.anomaly_detector.detect(df_for_quality)

        # 2. Tabular Drift Detection (if reference provided)
        drift_report = None
        if current_df is not None and reference_df is not None:
            drift_report = self.drift_detector.detect(reference_df, current_df)

        # 3. Time Series Detection (if timeseries inputs provided)
        ts_report: Optional[TimeSeriesReport] = None
        if timeseries_current is not None and timeseries_reference is not None:
            ts_detector = TimeSeriesDetector()
            ts_report = ts_detector.detect(
                current_data=timeseries_current, reference_data=timeseries_reference
            )

        # 4. Text Drift Detection (if text inputs provided)
        text_report: Optional[TextDriftReport] = None
        if text_current is not None and text_reference is not None:
            text_detector = TextDriftDetector()
            text_report = text_detector.detect(
                current_data=text_current, reference_data=text_reference
            )

        # Aggregate findings
        all_findings: list[Finding] = []
        if current_df is not None:
            all_findings.extend(quality_report.findings)
            all_findings.extend(anomaly_report.findings)
        if drift_report is not None:
            all_findings.extend(drift_report.findings)
        if ts_report is not None:
            all_findings.extend(ts_report.findings)
        if text_report is not None:
            all_findings.extend(text_report.findings)

        # Overall Status Gate
        has_critical = any(f.severity == Severity.CRITICAL for f in all_findings)
        has_high = any(f.severity == Severity.HIGH for f in all_findings)
        has_medium = any(f.severity == Severity.MEDIUM for f in all_findings)

        tabular_failed = current_df is not None and not quality_report.passed
        dataset_drift_failed = drift_report is not None and drift_report.dataset_drift
        ts_failed = ts_report is not None and ts_report.is_drift
        text_failed = text_report is not None and text_report.is_drift

        if has_critical or tabular_failed or dataset_drift_failed or ts_failed or text_failed:
            overall_status = "FAIL"
        elif has_high or has_medium:
            overall_status = "WARN"
        else:
            overall_status = "PASS"

        return DataSentryReport(
            report_id=report_id,
            timestamp=now_iso,
            dataset_name=dataset_name,
            quality=quality_report,
            anomaly=anomaly_report,
            drift=drift_report,
            timeseries=ts_report,
            text=text_report,
            all_findings=all_findings,
            overall_status=overall_status,
        )
