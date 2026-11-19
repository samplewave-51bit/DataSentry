"""
DataSentry Evaluation - Baselines Suite

Provides baseline detectors for benchmark comparisons:
1. Naive KS-test baseline
2. Great Expectations baseline
3. Evidently baseline
4. DataSentry Engine baseline
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
from scipy import stats

from datasentry.core.engine import DataSentryEngine


class BaseBenchmarkDetector:
    """Standardized wrapper for benchmark comparison tools."""

    name: str = "base"

    def run(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs detection and returns standardized dictionary:
        {
            "detector": str,
            "detected_fault": bool,
            "latency_ms": float,
            "details": dict
        }
        """
        raise NotImplementedError


class NaiveKSBaseline(BaseBenchmarkDetector):
    """Naive baseline running only two-sample Kolmogorov-Smirnov test on numeric features."""

    name: str = "naive_ks"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def run(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        detected = False
        drifted_cols = []

        if reference_df is not None:
            common_cols = [c for c in reference_df.columns if c in current_df.columns]
            for col in common_cols:
                ref_s = reference_df[col].dropna()
                cur_s = current_df[col].dropna()
                if pd.api.types.is_numeric_dtype(ref_s) and pd.api.types.is_numeric_dtype(cur_s):
                    if len(ref_s) > 5 and len(cur_s) > 5:
                        ks_res = stats.ks_2samp(ref_s, cur_s)
                        if ks_res.pvalue < self.alpha and ks_res.statistic > 0.15:
                            detected = True
                            drifted_cols.append(col)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "detector": self.name,
            "detected_fault": detected,
            "latency_ms": latency_ms,
            "details": {"drifted_columns": drifted_cols},
        }


class GreatExpectationsBaseline(BaseBenchmarkDetector):
    """Rule-based expectation baseline modeled after Great Expectations suites."""

    name: str = "great_expectations"

    def __init__(self, null_threshold: float = 0.15):
        self.null_threshold = null_threshold

    def run(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        detected = False
        failures = []

        # 1. Expect table columns to match schema
        if schema:
            expected_cols = set(schema.keys())
            found_cols = set(current_df.columns)
            if expected_cols != found_cols:
                detected = True
                failures.append(f"Column mismatch: missing={expected_cols - found_cols}, extra={found_cols - expected_cols}")

            for col, expected_type in schema.items():
                if col in current_df.columns:
                    actual_num = pd.api.types.is_numeric_dtype(current_df[col])
                    exp_num = expected_type in ("numeric", "float", "int")
                    if actual_num != exp_num:
                        detected = True
                        failures.append(f"Type mismatch on '{col}'")

        # 2. Expect column values to not be null beyond threshold
        for col in current_df.columns:
            null_ratio = float(current_df[col].isna().mean())
            if null_ratio > self.null_threshold:
                detected = True
                failures.append(f"Null ratio exceeded on '{col}': {null_ratio:.2%}")

        # 3. Expect values to be between reference min & max bounds
        if reference_df is not None:
            for col in reference_df.columns:
                if col in current_df.columns and pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(current_df[col]):
                    ref_min = float(reference_df[col].min())
                    ref_max = float(reference_df[col].max())
                    ref_range = ref_max - ref_min
                    cur_clean = current_df[col].dropna()
                    if len(cur_clean) > 0:
                        # Allow slight 1.5x margin
                        violating = (cur_clean < (ref_min - 1.5 * ref_range)) | (cur_clean > (ref_max + 1.5 * ref_range))
                        if violating.sum() > max(1, 0.02 * len(cur_clean)):
                            detected = True
                            failures.append(f"Out of boundary on '{col}'")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "detector": self.name,
            "detected_fault": detected,
            "latency_ms": latency_ms,
            "details": {"failures": failures},
        }


class EvidentlyBaseline(BaseBenchmarkDetector):
    """Evidently data quality and drift baseline."""

    name: str = "evidently"

    def __init__(self, drift_share_threshold: float = 0.33):
        self.drift_share_threshold = drift_share_threshold

    def run(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        detected = False
        details = {}

        # Check if evidently package can be imported directly
        try:
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset
            from evidently.report import Report

            # Sensible enterprise configuration: evaluate both Data Quality and Data Drift
            metrics = [DataQualityPreset()]
            if reference_df is not None:
                metrics.append(DataDriftPreset(drift_share=self.drift_share_threshold))
            report = Report(metrics=metrics)
            report.run(reference_data=reference_df, current_data=current_df)
            res = report.as_dict()

            # Inspect quality issues and drift metrics
            for m in res.get("metrics", []):
                metric_res = m.get("result", {})
                if "dataset_drift" in metric_res and metric_res["dataset_drift"]:
                    detected = True
                    details["drift_share"] = metric_res.get("drift_share", 0.0)
                # Check for severe missingness or schema issues
                if "number_of_missing_values" in metric_res and metric_res["number_of_missing_values"] > 0.20 * len(current_df):
                    detected = True
        except Exception:
            # Algorithmic fallback replicating Evidently's DataQuality + DataDrift presets
            if reference_df is not None:
                # 1. Schema columns check
                ref_cols = set(reference_df.columns)
                cur_cols = set(current_df.columns)
                if ref_cols != cur_cols:
                    detected = True
                    details["schema_mismatch"] = True

                # 2. Statistical drift (KS test with Bonferroni correction)
                drifted_cols = []
                common_cols = [c for c in reference_df.columns if c in current_df.columns]
                alpha_corr = 0.05 / max(1, len(common_cols))
                for col in common_cols:
                    if pd.api.types.is_numeric_dtype(reference_df[col]) and pd.api.types.is_numeric_dtype(current_df[col]):
                        ks_res = stats.ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
                        if ks_res.pvalue < alpha_corr and ks_res.statistic > 0.15:
                            drifted_cols.append(col)
                drift_share = len(drifted_cols) / max(1, len(common_cols))
                if drift_share >= self.drift_share_threshold:
                    detected = True
                details["drift_share"] = drift_share
                details["drifted_columns"] = drifted_cols

            # 3. Quality: Duplicate rows & missingness spikes
            if current_df.duplicated().mean() > 0.10 or (current_df.isna().mean().max() > 0.30):
                detected = True

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "detector": self.name,
            "detected_fault": detected,
            "latency_ms": latency_ms,
            "details": details,
        }


class DataSentryEngineBaseline(BaseBenchmarkDetector):
    """DataSentry Multi-Axis Engine Benchmark Wrapper."""

    name: str = "datasentry"

    def __init__(self, contamination: float = 0.05):
        self.contamination = contamination

    def run(
        self,
        current_df: pd.DataFrame,
        reference_df: Optional[pd.DataFrame] = None,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        engine = DataSentryEngine(
            schema=schema,
            contamination=self.contamination,
        )
        report = engine.scan(current_df=current_df, reference_df=reference_df)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        detected = report.overall_status in ("FAIL", "WARN")
        return {
            "detector": self.name,
            "detected_fault": detected,
            "latency_ms": latency_ms,
            "details": {
                "overall_status": report.overall_status,
                "findings_count": len(report.all_findings),
                "quality_passed": report.quality.passed,
                "anomaly_count": report.anomaly.anomaly_count,
                "dataset_drift": report.drift.dataset_drift if report.drift else False,
            },
        }
