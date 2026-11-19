"""
DataSentry Deep Learning Component - PyTorch Tabular Domain Classifier

Trains a neural network to discriminate Reference (0) vs. Current (1) batches.
Outputs the held-out ROC AUC as a learned dataset drift score.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from datasentry.schemas import (
    AxisType,
    DriftMetric,
    Finding,
    Severity,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TabularDomainClassifier(nn.Module):
    """Binary classifier predicting domain source (Reference=0, Current=1)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout_rate: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainClassifierDriftScorer:
    """Computes learned dataset-level distribution shift via PyTorch Domain Classifier AUC."""

    def __init__(
        self,
        hidden_dim: int = 32,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        auc_threshold: float = 0.65,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.auc_threshold = auc_threshold
        self.random_state = random_state

    def score(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> Tuple[float, bool, DriftMetric, List[Finding]]:
        """
        Trains domain classifier using 3-fold cross validation.
        Returns: (auc_score, is_drift, drift_metric, findings)
        """
        set_seed(self.random_state)

        # Select shared numeric features
        shared_cols = [
            c for c in reference_df.select_dtypes(include=[np.number]).columns
            if c in current_df.columns and pd.api.types.is_numeric_dtype(current_df[c])
        ]
        if len(shared_cols) < 1 or len(reference_df) < 10 or len(current_df) < 10:
            metric = DriftMetric(
                feature="all_features",
                method="pytorch_domain_classifier",
                statistic=0.50,
                threshold=self.auc_threshold,
                drifted=False,
            )
            return 0.50, False, metric, []

        ref_clean = reference_df[shared_cols].dropna()
        cur_clean = current_df[shared_cols].dropna()

        # Combine with domain labels: Ref=0, Cur=1
        scaler = StandardScaler()
        x_all = np.vstack([ref_clean.values, cur_clean.values])
        x_scaled = scaler.fit_transform(x_all)
        y_all = np.array([0] * len(ref_clean) + [1] * len(cur_clean), dtype=np.float32)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        fold_aucs = []

        for train_idx, val_idx in skf.split(x_scaled, y_all):
            x_train, y_train = x_scaled[train_idx], y_all[train_idx]
            x_val, y_val = x_scaled[val_idx], y_all[val_idx]

            model = TabularDomainClassifier(
                input_dim=len(shared_cols), hidden_dim=self.hidden_dim
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()

            train_ds = TensorDataset(
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
            )
            train_loader = DataLoader(
                train_ds, batch_size=min(self.batch_size, len(train_ds)), shuffle=True
            )

            model.train()
            for _ in range(self.epochs):
                for x_b, y_b in train_loader:
                    optimizer.zero_grad()
                    out = model(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(x_val, dtype=torch.float32)).squeeze(1)
                val_probs = torch.sigmoid(val_logits).numpy()
                try:
                    fold_auc = float(roc_auc_score(y_val, val_probs))
                    fold_aucs.append(fold_auc)
                except ValueError:
                    fold_aucs.append(0.50)

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.50
        is_drift = bool(mean_auc >= self.auc_threshold)

        findings: List[Finding] = []
        if is_drift:
            findings.append(
                Finding(
                    finding_id="FND-DRFT-DL-001",
                    axis=AxisType.DRIFT,
                    severity=Severity.CRITICAL if mean_auc > 0.85 else Severity.HIGH,
                    metric_name="domain_classifier_auc",
                    metric_value=mean_auc,
                    threshold=self.auc_threshold,
                    description=(
                        f"Learned distribution shift detected by PyTorch Domain Classifier "
                        f"(ROC AUC={mean_auc:.3f} >= threshold {self.auc_threshold:.2f})."
                    ),
                )
            )

        metric = DriftMetric(
            feature="dataset_level_features",
            method="pytorch_domain_classifier",
            statistic=mean_auc,
            threshold=self.auc_threshold,
            drifted=is_drift,
            details={"fold_aucs": [round(a, 4) for a in fold_aucs], "features": shared_cols},
        )
        return mean_auc, is_drift, metric, findings
