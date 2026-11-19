"""
DataSentry Core - Axis 3: Dataset-Level Drift Detection

Compares Reference (baseline) vs. Current incoming batches using
Kolmogorov-Smirnov, Wasserstein, PSI, and Chi-Square tests.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from datasentry.schemas import (
    AxisType,
    DriftMetric,
    DriftReport,
    Finding,
    Severity,
)


def calculate_psi(
    expected: np.ndarray, actual: np.ndarray, num_bins: int = 10
) -> float:
    """Calculate Population Stability Index (PSI) between two 1D numeric arrays."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected
    percentiles = np.linspace(0, 100, num_bins + 1)
    try:
        bins = np.percentile(expected, percentiles)
    except Exception:
        return 0.0

    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    # Expand outer bounds
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Add small epsilon to prevent zero division
    eps = 1e-4
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(bins))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(bins))

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.maximum(0.0, psi_val))


class DriftDetector:
    """Detects feature-level and dataset-level distribution shift."""

    def __init__(
        self,
        p_val_threshold: float = 0.05,
        psi_threshold: float = 0.20,
        dataset_drift_threshold: float = 0.33,
    ):
        """
        Args:
            p_val_threshold: Significance level for KS & Chi-Square tests (default: 0.05).
            psi_threshold: PSI value indicating significant shift (default: 0.20).
            dataset_drift_threshold: Ratio of drifted features required to declare dataset-level drift.
        """
        self.p_val_threshold = p_val_threshold
        self.psi_threshold = psi_threshold
        self.dataset_drift_threshold = dataset_drift_threshold

    def detect(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> DriftReport:
        """Compare reference and current batches for distribution drift."""
        shared_cols = [c for c in reference_df.columns if c in current_df.columns]
        if not shared_cols:
            return DriftReport(
                total_features=0,
                drifted_features_count=0,
                drift_share=0.0,
                dataset_drift=False,
                drift_threshold_share=self.dataset_drift_threshold,
                metrics={},
                findings=[],
            )

        metrics: Dict[str, DriftMetric] = {}
        drifted_features: List[str] = []
        findings: List[Finding] = []

        for col in shared_cols:
            ref_col = reference_df[col].dropna()
            cur_col = current_df[col].dropna()

            if len(ref_col) < 5 or len(cur_col) < 5:
                continue

            if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
                metric = self._detect_numeric_drift(col, ref_col.to_numpy(), cur_col.to_numpy())
            else:
                metric = self._detect_categorical_drift(col, ref_col, cur_col)

            metrics[col] = metric
            if metric.drifted:
                drifted_features.append(col)
                stat_val = metric.statistic
                p_val = metric.p_value
                p_str = f", p-value={p_val:.4f}" if p_val is not None else ""
                findings.append(
                    Finding(
                        finding_id=f"FND-DRFT-{len(findings) + 1:03d}",
                        axis=AxisType.DRIFT,
                        severity=Severity.HIGH if stat_val > 0.30 else Severity.MEDIUM,
                        column=col,
                        metric_name=f"{metric.method}_drift",
                        metric_value=stat_val,
                        threshold=metric.threshold,
                        description=(
                            f"Drift detected in feature '{col}' using {metric.method} "
                            f"(statistic={stat_val:.4f}{p_str}, threshold={metric.threshold})."
                        ),
                    )
                )

        total_feats = len(metrics)
        drift_count = len(drifted_features)
        drift_share = float(drift_count / total_feats) if total_feats > 0 else 0.0
        dataset_drift = drift_share >= self.dataset_drift_threshold

        if dataset_drift and drift_count > 0:
            findings.insert(
                0,
                Finding(
                    finding_id="FND-DRFT-DATASET",
                    axis=AxisType.DRIFT,
                    severity=Severity.CRITICAL if drift_share > 0.6 else Severity.HIGH,
                    metric_name="dataset_drift_share",
                    metric_value=drift_share,
                    threshold=self.dataset_drift_threshold,
                    description=(
                        f"Dataset-level drift detected: {drift_count}/{total_feats} features "
                        f"({drift_share:.1%}) drifted, exceeding threshold {self.dataset_drift_threshold:.1%}."
                    ),
                ),
            )

        return DriftReport(
            total_features=total_feats,
            drifted_features_count=drift_count,
            drift_share=drift_share,
            dataset_drift=dataset_drift,
            drift_threshold_share=self.dataset_drift_threshold,
            metrics=metrics,
            findings=findings,
        )

    def _detect_numeric_drift(
        self, col: str, ref: np.ndarray, cur: np.ndarray
    ) -> DriftMetric:
        # Two-sample Kolmogorov-Smirnov test
        ks_res = stats.ks_2samp(ref, cur)
        # Earth Mover's Distance (Wasserstein)
        wass_dist = float(stats.wasserstein_distance(ref, cur))
        # PSI
        psi_score = calculate_psi(ref, cur)

        # Drift flagged if KS test is significant and statistic > 0.1, or PSI > threshold
        is_ks_drift = bool(ks_res.pvalue < self.p_val_threshold and ks_res.statistic > 0.10)
        is_psi_drift = bool(psi_score > self.psi_threshold)
        drifted = is_ks_drift or is_psi_drift

        return DriftMetric(
            feature=col,
            method="kolmogorov_smirnov",
            statistic=float(ks_res.statistic),
            p_value=float(ks_res.pvalue),
            threshold=float(self.p_val_threshold),
            drifted=drifted,
            details={
                "wasserstein_distance": wass_dist,
                "psi": psi_score,
                "ks_drift": is_ks_drift,
                "psi_drift": is_psi_drift,
            },
        )

    def _detect_categorical_drift(
        self, col: str, ref_s: pd.Series, cur_s: pd.Series
    ) -> DriftMetric:
        # Align category counts
        ref_counts = ref_s.value_counts()
        cur_counts = cur_s.value_counts()
        all_cats = list(set(ref_counts.index).union(set(cur_counts.index)))

        ref_aligned = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
        cur_aligned = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)

        # Normalize to frequencies for chi-square
        ref_freq = (ref_aligned + 1e-4) / (ref_aligned.sum() + 1e-4 * len(all_cats))
        cur_freq = (cur_aligned + 1e-4) / (cur_aligned.sum() + 1e-4 * len(all_cats))

        # Expected frequencies scaled to current sample size
        expected = ref_freq * len(cur_s)
        actual = cur_aligned

        try:
            chi2_stat, p_val = stats.chisquare(f_obs=actual, f_exp=expected)
            drifted = bool(p_val < self.p_val_threshold)
        except Exception:
            chi2_stat, p_val = 0.0, 1.0
            drifted = False

        # Compute categorical PSI
        cat_psi = float(np.sum((cur_freq - ref_freq) * np.log(cur_freq / ref_freq)))
        if cat_psi > self.psi_threshold:
            drifted = True

        return DriftMetric(
            feature=col,
            method="chi_square",
            statistic=float(chi2_stat),
            p_value=float(p_val),
            threshold=float(self.p_val_threshold),
            drifted=drifted,
            details={"categorical_psi": cat_psi, "distinct_categories": len(all_cats)},
        )
