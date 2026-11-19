"""
DataSentry Core - Axis 2: Record-Level Anomaly Detection

Implements rule-based (IQR/Z-score) and statistical (Isolation Forest, Mahalanobis)
anomaly detection with explainable feature contribution vectors.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from datasentry.schemas import (
    AnomalyRecord,
    AnomalyReport,
    AxisType,
    Finding,
    Severity,
)


class AnomalyDetector:
    """Record-level anomaly detector for tabular datasets."""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        """
        Args:
            method: 'isolation_forest', 'iqr', or 'mahalanobis'.
            contamination: Expected proportion of outliers in the data.
            random_state: Seed for reproducibility.
        """
        self.method = method.lower()
        self.contamination = contamination
        self.random_state = random_state

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        """Detect anomalous rows in numeric features of df."""
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty or len(numeric_df) < 5:
            return AnomalyReport(
                total_evaluated=len(df),
                anomaly_count=0,
                anomaly_percentage=0.0,
                method=self.method,
                anomalies=[],
                findings=[],
            )

        if self.method == "iqr":
            records = self._detect_iqr(df, numeric_df)
        elif self.method == "mahalanobis":
            records = self._detect_mahalanobis(df, numeric_df)
        else:
            records = self._detect_isolation_forest(df, numeric_df)

        anomaly_count = sum(1 for r in records if r.is_anomaly)
        anomaly_pct = float(anomaly_count / len(df)) if len(df) > 0 else 0.0

        findings: List[Finding] = []
        if anomaly_pct > self.contamination * 1.5 and anomaly_count > 0:
            findings.append(
                Finding(
                    finding_id="FND-ANOM-001",
                    axis=AxisType.ANOMALY,
                    severity=Severity.HIGH if anomaly_pct > 0.15 else Severity.MEDIUM,
                    metric_name=f"{self.method}_anomaly_rate",
                    metric_value=anomaly_pct,
                    threshold=self.contamination,
                    description=(
                        f"Elevated record anomaly rate ({anomaly_pct:.2%}, {anomaly_count} rows) "
                        f"exceeds expected contamination threshold ({self.contamination:.2%})."
                    ),
                )
            )

        return AnomalyReport(
            total_evaluated=len(df),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_pct,
            method=self.method,
            anomalies=records,
            findings=findings,
        )

    def _detect_isolation_forest(
        self, original_df: pd.DataFrame, numeric_df: pd.DataFrame
    ) -> List[AnomalyRecord]:
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        preds = model.fit_predict(numeric_df)
        scores = -model.score_samples(numeric_df)  # higher = more anomalous

        # Standardize features to compute feature contribution (deviation from median)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(numeric_df)
        feature_deviations = np.abs(scaled)

        records: List[AnomalyRecord] = []
        for i, (orig_idx, _) in enumerate(numeric_df.iterrows()):
            is_anom = bool(preds[i] == -1)
            score = float(scores[i])

            # Top 3 contributing features
            top_feature_indices = np.argsort(feature_deviations[i])[-3:][::-1]
            contributions: Dict[str, float] = {
                numeric_df.columns[feat_idx]: float(feature_deviations[i, feat_idx])
                for feat_idx in top_feature_indices
            }

            rationale = ""
            if is_anom:
                top_items = [f"{k} (dev={v:.2f})" for k, v in contributions.items()]
                rationale = f"Outlier detected by Isolation Forest. Highest deviations: {', '.join(top_items)}"

            records.append(
                AnomalyRecord(
                    row_index=int(orig_idx),
                    anomaly_score=score,
                    is_anomaly=is_anom,
                    contributing_features=contributions,
                    rationale=rationale,
                )
            )
        return records

    def _detect_iqr(
        self, original_df: pd.DataFrame, numeric_df: pd.DataFrame
    ) -> List[AnomalyRecord]:
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        # Prevent division by zero
        iqr_clean = iqr.replace(0, 1e-6)

        lower_bound = q1 - 1.5 * iqr_clean
        upper_bound = q3 + 1.5 * iqr_clean

        is_below = numeric_df < lower_bound
        is_above = numeric_df > upper_bound
        violating = is_below | is_above

        records: List[AnomalyRecord] = []
        for orig_idx, row in numeric_df.iterrows():
            row_viols = violating.loc[orig_idx]
            num_viols = int(row_viols.sum())
            is_anom = num_viols > 0

            contributions: Dict[str, float] = {}
            for col in numeric_df.columns:
                if row_viols[col]:
                    dist = max(
                        float(lower_bound[col] - row[col]),
                        float(row[col] - upper_bound[col]),
                    )
                    contributions[col] = float(dist / iqr_clean[col])

            score = float(num_viols / len(numeric_df.columns))
            rationale = (
                f"Violated IQR bounds on {len(contributions)} features: {list(contributions.keys())}"
                if is_anom
                else ""
            )

            records.append(
                AnomalyRecord(
                    row_index=int(orig_idx),
                    anomaly_score=score,
                    is_anomaly=is_anom,
                    contributing_features=contributions,
                    rationale=rationale,
                )
            )
        return records

    def _detect_mahalanobis(
        self, original_df: pd.DataFrame, numeric_df: pd.DataFrame
    ) -> List[AnomalyRecord]:
        vals = numeric_df.values
        mean = np.mean(vals, axis=0)
        cov = np.cov(vals, rowvar=False)

        # Apply shrinkage regularization to guarantee invertibility
        cov_reg = cov + 1e-5 * np.eye(cov.shape[0])
        try:
            inv_cov = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov_reg)

        diff = vals - mean
        # Mahalanobis distance squared: d^2 = (x - mu)^T * Sigma^-1 * (x - mu)
        md2 = np.sum(diff @ inv_cov * diff, axis=1)
        distances = np.sqrt(np.maximum(0, md2))

        # Threshold based on chi-square distribution quantile
        from scipy.stats import chi2

        p_val_thresh = 1.0 - self.contamination
        cutoff = np.sqrt(chi2.ppf(p_val_thresh, df=vals.shape[1]))

        records: List[AnomalyRecord] = []
        for i, (orig_idx, _) in enumerate(numeric_df.iterrows()):
            dist = float(distances[i])
            is_anom = dist > cutoff

            # Feature contributions via standardized z-score
            std = np.std(vals, axis=0) + 1e-6
            z_scores = np.abs(diff[i] / std)
            top_indices = np.argsort(z_scores)[-3:][::-1]
            contributions = {
                numeric_df.columns[idx]: float(z_scores[idx]) for idx in top_indices
            }

            rationale = (
                f"Mahalanobis distance {dist:.2f} exceeds threshold {cutoff:.2f}"
                if is_anom
                else ""
            )

            records.append(
                AnomalyRecord(
                    row_index=int(orig_idx),
                    anomaly_score=dist,
                    is_anomaly=is_anom,
                    contributing_features=contributions,
                    rationale=rationale,
                )
            )
        return records
