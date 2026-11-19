"""
DataSentry Core Schemas

Strict Pydantic models for reporting data-quality validation,
record-level anomaly detection, and dataset drift.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AxisType(str, Enum):
    QUALITY = "quality"
    ANOMALY = "anomaly"
    DRIFT = "drift"
    TIMESERIES = "timeseries"
    TEXT = "text"


class Finding(BaseModel):
    """Atomic diagnostic finding that can be cited by IDs (e.g., FND-001)."""
    finding_id: str = Field(description="Unique identifier for citation (e.g. FND-001)")
    axis: AxisType = Field(description="Analytical axis that surfaced the finding")
    severity: Severity = Field(description="Severity rating")
    column: Optional[str] = Field(default=None, description="Feature column involved, if applicable")
    metric_name: str = Field(description="Name of the evaluated metric (e.g., ks_statistic, null_ratio)")
    metric_value: float = Field(description="Empirically computed value")
    threshold: float = Field(description="Threshold that was breached")
    description: str = Field(description="Clear explanation of the observed issue")


class ColumnQualityMetric(BaseModel):
    column: str
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    is_constant: bool = False
    out_of_bounds_count: int = 0
    violations: List[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    passed: bool
    total_rows: int
    total_columns: int
    duplicate_rows: int
    duplicate_percentage: float
    schema_matched: bool
    schema_violations: List[str] = Field(default_factory=list)
    column_metrics: Dict[str, ColumnQualityMetric] = Field(default_factory=dict)
    findings: List[Finding] = Field(default_factory=list)


class AnomalyRecord(BaseModel):
    row_index: int
    anomaly_score: float
    is_anomaly: bool
    contributing_features: Dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class AnomalyReport(BaseModel):
    total_evaluated: int
    anomaly_count: int
    anomaly_percentage: float
    method: str
    anomalies: List[AnomalyRecord] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)


class DriftMetric(BaseModel):
    feature: str
    method: str
    statistic: float
    p_value: Optional[float] = None
    threshold: float
    drifted: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    total_features: int
    drifted_features_count: int
    drift_share: float
    dataset_drift: bool
    drift_threshold_share: float = 0.33
    metrics: Dict[str, DriftMetric] = Field(default_factory=dict)
    findings: List[Finding] = Field(default_factory=list)


class TimeSeriesPoint(BaseModel):
    timestamp: str
    actual: float
    predicted: float
    lower_bound: float
    upper_bound: float
    residual: float
    is_anomaly: bool


class TimeSeriesReport(BaseModel):
    total_points: int
    anomaly_count: int
    anomaly_percentage: float
    calibrated_coverage: float
    empirical_coverage: float
    interval_alpha: float = 0.05
    mean_absolute_error: float
    is_drift: bool
    forecaster_model: str = "seasonal_residual_forecaster"
    baseline_model: str = "arima_baseline"
    baseline_coverage: float = 0.0
    baseline_mae: float = 0.0
    points: List[TimeSeriesPoint] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)


class TextDriftReport(BaseModel):
    total_reference_docs: int
    total_current_docs: int
    embedding_model: str
    mmd_score: float
    mmd_threshold: float = 0.05
    domain_classifier_auc: float
    auc_threshold: float = 0.65
    centroid_cosine_distance: float
    cosine_threshold: float = 0.15
    is_drift: bool
    top_divergent_terms: List[str] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)


class DataSentryReport(BaseModel):
    """Consolidated DataSentry Diagnostic Report."""
    report_id: str
    timestamp: str
    dataset_name: str
    quality: QualityReport
    anomaly: AnomalyReport
    drift: Optional[DriftReport] = None
    timeseries: Optional[TimeSeriesReport] = None
    text: Optional[TextDriftReport] = None
    all_findings: List[Finding] = Field(default_factory=list)
    overall_status: str = Field(
        default="PASS",
        description="Overall pipeline gate decision: PASS, WARN, or FAIL"
    )
