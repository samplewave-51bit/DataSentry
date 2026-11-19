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
from datasentry.schemas import (
    DataSentryReport,
    Finding,
    Severity,
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
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        dataset_name: str = "dataset_batch",
    ) -> DataSentryReport:
        """Run all active diagnostic axes and compile report."""
        report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()

        # 1. Quality Validation
        quality_report = self.quality_validator.validate(current_df)

        # 2. Record Anomaly Detection
        anomaly_report = self.anomaly_detector.detect(current_df)

        # 3. Drift Detection (if reference provided)
        drift_report = None
        if reference_df is not None:
            drift_report = self.drift_detector.detect(reference_df, current_df)

        # Aggregate findings
        all_findings: list[Finding] = []
        all_findings.extend(quality_report.findings)
        all_findings.extend(anomaly_report.findings)
        if drift_report is not None:
            all_findings.extend(drift_report.findings)

        # Overall Status Gate
        has_critical = any(f.severity == Severity.CRITICAL for f in all_findings)
        has_high = any(f.severity == Severity.HIGH for f in all_findings)
        has_medium = any(f.severity == Severity.MEDIUM for f in all_findings)

        if has_critical or not quality_report.passed or (drift_report and drift_report.dataset_drift):
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
            all_findings=all_findings,
            overall_status=overall_status,
        )
