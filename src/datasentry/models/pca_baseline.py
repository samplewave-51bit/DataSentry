"""
DataSentry Classical Baseline - PCA Reconstruction Error Anomaly Detector

Projects data to linear subspace and computes reconstruction MSE
to serve as an ablation baseline against the non-linear PyTorch Autoencoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from datasentry.schemas import (
    AnomalyRecord,
    AnomalyReport,
    AxisType,
    Finding,
    Severity,
)


class PCAReconstructionDetector:
    """Anomaly detection using PCA linear subspace reconstruction error."""

    def __init__(self, n_components: Optional[int] = None, contamination: float = 0.05, random_state: int = 42):
        self.n_components = n_components
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.threshold: float = 0.0
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "PCAReconstructionDetector":
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty or len(numeric_df) < 5:
            raise ValueError("PCA detector requires at least 5 numeric rows.")

        self.feature_names = list(numeric_df.columns)
        scaled = self.scaler.fit_transform(numeric_df)

        n_comp = self.n_components or max(1, min(len(self.feature_names) - 1, 3))
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        transformed = self.pca.fit_transform(scaled)
        reconstructed = self.pca.inverse_transform(transformed)

        ref_errors = np.mean((scaled - reconstructed) ** 2, axis=1)
        self.threshold = float(np.percentile(ref_errors, 100.0 * (1.0 - self.contamination)))
        return self

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        if self.pca is None:
            raise RuntimeError("PCAReconstructionDetector must be fitted before detect().")

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty:
            return AnomalyReport(
                total_evaluated=len(df),
                anomaly_count=0,
                anomaly_percentage=0.0,
                method="pca_reconstruction",
                anomalies=[],
                findings=[],
            )

        aligned_df = numeric_df[[c for c in self.feature_names if c in numeric_df.columns]]
        scaled = self.scaler.transform(aligned_df)

        transformed = self.pca.transform(scaled)
        reconstructed = self.pca.inverse_transform(transformed)

        sq_diffs = (scaled - reconstructed) ** 2
        row_errors = np.mean(sq_diffs, axis=1)

        records: List[AnomalyRecord] = []
        for i, (orig_idx, _) in enumerate(aligned_df.iterrows()):
            score = float(row_errors[i])
            is_anom = bool(score > self.threshold)

            top_feat_idx = np.argsort(sq_diffs[i])[-3:][::-1]
            contributions: Dict[str, float] = {
                self.feature_names[f_idx]: float(sq_diffs[i, f_idx])
                for f_idx in top_feat_idx
            }

            records.append(
                AnomalyRecord(
                    row_index=int(orig_idx),
                    anomaly_score=score,
                    is_anomaly=is_anom,
                    contributing_features=contributions,
                    rationale=f"PCA reconstruction error {score:.3f} > threshold {self.threshold:.3f}" if is_anom else "",
                )
            )

        anomaly_count = sum(1 for r in records if r.is_anomaly)
        anomaly_pct = float(anomaly_count / len(df)) if len(df) > 0 else 0.0

        findings: List[Finding] = []
        if anomaly_pct > self.contamination * 1.5 and anomaly_count > 0:
            findings.append(
                Finding(
                    finding_id="FND-ANOM-PCA-001",
                    axis=AxisType.ANOMALY,
                    severity=Severity.HIGH if anomaly_pct > 0.15 else Severity.MEDIUM,
                    metric_name="pca_reconstruction_anomaly_rate",
                    metric_value=anomaly_pct,
                    threshold=self.contamination,
                    description=f"Elevated PCA reconstruction anomalies: {anomaly_count} rows ({anomaly_pct:.1%}).",
                )
            )

        return AnomalyReport(
            total_evaluated=len(df),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_pct,
            method="pca_reconstruction",
            anomalies=records,
            findings=findings,
        )
