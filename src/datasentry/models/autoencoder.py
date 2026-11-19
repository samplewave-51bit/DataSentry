"""
DataSentry Deep Learning Component - PyTorch Tabular Autoencoder

Implements a deep autoencoder for tabular reconstruction anomaly detection
with explainable feature-level attribution and deterministic seed control.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from datasentry.schemas import (
    AnomalyRecord,
    AnomalyReport,
    AxisType,
    Finding,
    Severity,
)


def set_seed(seed: int = 42):
    """Ensures 100% deterministic reproducibility across PyTorch and NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularAutoencoder(nn.Module):
    """Multi-layer fully-connected autoencoder with bottleneck latent layer."""

    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderAnomalyDetector:
    """Tabular anomaly detector using PyTorch Autoencoder reconstruction error."""

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 32,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.005,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.contamination = contamination
        self.random_state = random_state

        self.model: Optional[TabularAutoencoder] = None
        self.scaler = StandardScaler()
        self.threshold: float = 0.0
        self.feature_names: List[str] = []

    def fit(self, df: pd.DataFrame) -> "AutoencoderAnomalyDetector":
        """Fits autoencoder on reference clean tabular data."""
        set_seed(self.random_state)
        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty or len(numeric_df) < 10:
            raise ValueError("Autoencoder requires at least 10 rows with numeric features.")

        self.feature_names = list(numeric_df.columns)
        scaled_vals = self.scaler.fit_transform(numeric_df)
        x_tensor = torch.tensor(scaled_vals, dtype=torch.float32)

        input_dim = len(self.feature_names)
        actual_latent = min(self.latent_dim, max(2, input_dim // 2))
        self.model = TabularAutoencoder(
            input_dim=input_dim,
            latent_dim=actual_latent,
            hidden_dim=self.hidden_dim,
        )

        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for batch in loader:
                inputs = batch[0]
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

        # Calibrate threshold on reference reconstruction errors
        self.model.eval()
        with torch.no_grad():
            ref_preds = self.model(x_tensor).numpy()
            ref_errors = np.mean((scaled_vals - ref_preds) ** 2, axis=1)
            # Threshold at 1 - contamination quantile
            self.threshold = float(np.percentile(ref_errors, 100.0 * (1.0 - self.contamination)))

        return self

    def detect(self, df: pd.DataFrame) -> AnomalyReport:
        """Evaluates reconstruction error on incoming batch."""
        if self.model is None:
            raise RuntimeError("AutoencoderAnomalyDetector must be fitted before detect().")

        numeric_df = df.select_dtypes(include=[np.number]).dropna()
        if numeric_df.empty:
            return AnomalyReport(
                total_evaluated=len(df),
                anomaly_count=0,
                anomaly_percentage=0.0,
                method="pytorch_autoencoder",
                anomalies=[],
                findings=[],
            )

        # Retain only features seen during training
        aligned_df = numeric_df[[c for c in self.feature_names if c in numeric_df.columns]]
        scaled_vals = self.scaler.transform(aligned_df)
        x_tensor = torch.tensor(scaled_vals, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(x_tensor).numpy()
            sq_diffs = (scaled_vals - reconstructed) ** 2
            row_errors = np.mean(sq_diffs, axis=1)

        records: List[AnomalyRecord] = []
        for i, (orig_idx, _) in enumerate(aligned_df.iterrows()):
            score = float(row_errors[i])
            is_anom = bool(score > self.threshold)

            # Top 3 features by reconstruction error
            top_feat_idx = np.argsort(sq_diffs[i])[-3:][::-1]
            contributions: Dict[str, float] = {
                self.feature_names[f_idx]: float(sq_diffs[i, f_idx])
                for f_idx in top_feat_idx
            }

            rationale = ""
            if is_anom:
                contrib_str = [f"{k} (err={v:.2f})" for k, v in contributions.items()]
                rationale = (
                    f"Autoencoder reconstruction error {score:.3f} > threshold {self.threshold:.3f}. "
                    f"Top error features: {', '.join(contrib_str)}"
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

        anomaly_count = sum(1 for r in records if r.is_anomaly)
        anomaly_pct = float(anomaly_count / len(df)) if len(df) > 0 else 0.0

        findings: List[Finding] = []
        if anomaly_pct > self.contamination * 1.5 and anomaly_count > 0:
            findings.append(
                Finding(
                    finding_id="FND-ANOM-AE-001",
                    axis=AxisType.ANOMALY,
                    severity=Severity.HIGH if anomaly_pct > 0.15 else Severity.MEDIUM,
                    metric_name="autoencoder_reconstruction_anomaly_rate",
                    metric_value=anomaly_pct,
                    threshold=self.contamination,
                    description=(
                        f"PyTorch Autoencoder detected elevated anomaly rate ({anomaly_pct:.1%}, {anomaly_count} rows) "
                        f"violating calibrated reconstruction threshold ({self.threshold:.3f})."
                    ),
                )
            )

        return AnomalyReport(
            total_evaluated=len(df),
            anomaly_count=anomaly_count,
            anomaly_percentage=anomaly_pct,
            method="pytorch_autoencoder",
            anomalies=records,
            findings=findings,
        )
