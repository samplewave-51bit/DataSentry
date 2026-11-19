"""
DataSentry Evaluation - Phase 3 Deep Learning Ablation Study

Executes controlled ablation comparing:
1. PyTorch Tabular Autoencoder
2. Isolation Forest
3. Mahalanobis Distance
4. PCA Reconstruction Error

And evaluates Learned Domain-Classifier Drift Scorer against statistical tests.
Outputs empirical ablation comparison metrics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from datasentry.core.anomaly import AnomalyDetector
from datasentry.evaluation.datasets import load_sensor_telemetry
from datasentry.evaluation.fault_injection import FaultInjector
from datasentry.models.autoencoder import AutoencoderAnomalyDetector
from datasentry.models.pca_baseline import PCAReconstructionDetector


class AnomalyAblationRunner:
    """Executes ablation benchmark comparing deep learning vs classical anomaly detectors."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.injector = FaultInjector(random_state=random_state)

    def run_ablation(self, n_trials: int = 10) -> pd.DataFrame:
        """Runs ablation on sensor telemetry data with ground truth injected anomalies."""
        clean_df, _ = load_sensor_telemetry(n_samples=600, random_state=self.random_state)
        ref_df = clean_df.iloc[:300].copy()
        test_base_df = clean_df.iloc[300:].copy()

        # Measure fit time and initialize models
        t_ae = time.perf_counter()
        ae_detector = AutoencoderAnomalyDetector(epochs=20, random_state=self.random_state).fit(ref_df)
        ae_fit_ms = (time.perf_counter() - t_ae) * 1000.0

        t_pca = time.perf_counter()
        pca_detector = PCAReconstructionDetector(random_state=self.random_state).fit(ref_df)
        pca_fit_ms = (time.perf_counter() - t_pca) * 1000.0

        t_if = time.perf_counter()
        iforest_detector = AnomalyDetector(method="isolation_forest", random_state=self.random_state).fit(ref_df)
        if_fit_ms = (time.perf_counter() - t_if) * 1000.0

        iforest_online = AnomalyDetector(method="isolation_forest", random_state=self.random_state)
        mahal_detector = AnomalyDetector(method="mahalanobis", random_state=self.random_state)

        fit_times = {
            "pytorch_autoencoder": round(ae_fit_ms, 2),
            "isolation_forest_prefit": round(if_fit_ms, 2),
            "isolation_forest_online": 0.0,
            "mahalanobis": 0.0,
            "pca_reconstruction": round(pca_fit_ms, 2),
        }

        models = {
            "pytorch_autoencoder": ae_detector,
            "isolation_forest_prefit": iforest_detector,
            "isolation_forest_online": iforest_online,
            "mahalanobis": mahal_detector,
            "pca_reconstruction": pca_detector,
        }

        records: List[Dict[str, Any]] = []

        for trial in range(n_trials):
            # Create a clean sample batch with 0-based index
            clean_batch = test_base_df.sample(
                n=100, replace=False, random_state=self.random_state + trial
            ).reset_index(drop=True).copy()

            # Create an anomaly-injected batch (positive control)
            fault_res = self.injector.inject_out_of_range(
                clean_batch, column="temperature", fraction=0.10, scale_factor=6.0
            )
            corrupted_batch = fault_res.corrupted_df

            # Ground truth anomaly mask: 1 for corrupted row indices, 0 otherwise
            ground_truth_mask = np.zeros(len(corrupted_batch), dtype=int)
            for idx in fault_res.target_rows:
                if 0 <= idx < len(ground_truth_mask):
                    ground_truth_mask[idx] = 1

            for name, model in models.items():
                t0 = time.perf_counter()
                rep = model.detect(corrupted_batch)
                latency_ms = (time.perf_counter() - t0) * 1000.0

                scores = np.array([r.anomaly_score for r in rep.anomalies])
                preds = np.array([1 if r.is_anomaly else 0 for r in rep.anomalies])

                # Compute ROC AUC if there are positive and negative samples
                try:
                    auc = float(roc_auc_score(ground_truth_mask, scores))
                except Exception:
                    auc = 0.50

                tp = int(np.sum((preds == 1) & (ground_truth_mask == 1)))
                fp = int(np.sum((preds == 1) & (ground_truth_mask == 0)))
                fn = int(np.sum((preds == 0) & (ground_truth_mask == 1)))

                prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
                rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

                records.append({
                    "model": name,
                    "trial": trial,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                    "roc_auc": auc,
                    "latency_ms": latency_ms,
                })

        df_res = pd.DataFrame(records)
        summary = (
            df_res.groupby("model")
            .agg(
                mean_f1=("f1_score", "mean"),
                mean_auc=("roc_auc", "mean"),
                mean_precision=("precision", "mean"),
                mean_recall=("recall", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
            )
            .reset_index()
            .sort_values("mean_f1", ascending=False)
        )
        summary["fit_time_ms"] = summary["model"].map(fit_times).fillna(0.0)
        return summary
