"""
Phase 3: Deep Learning Ablation & Learned Drift Benchmark Runner

Executes:
1. 4-way Tabular Anomaly Ablation: PyTorch Autoencoder vs Isolation Forest vs Mahalanobis vs PCA.
2. PyTorch Tabular Domain Classifier vs Statistical Drift Tests.
3. Outputs saved to outputs/ablation_results.csv and outputs/ablation_results.md.
"""

import os

from datasentry.evaluation.ablation import AnomalyAblationRunner
from datasentry.evaluation.datasets import load_sensor_telemetry
from datasentry.evaluation.fault_injection import FaultInjector
from datasentry.models.domain_classifier import DomainClassifierDriftScorer


def run_drift_ablation():
    print("\nEvaluating PyTorch Domain Classifier Drift Scorer vs Statistical Tests...")
    clean_df, _ = load_sensor_telemetry(n_samples=500, random_state=42)
    ref_df = clean_df.iloc[:250].copy()
    cur_clean = clean_df.iloc[250:].copy()

    # Shifted current batch
    injector = FaultInjector(random_state=42)
    cur_shifted = injector.inject_covariate_shift(cur_clean, column="temperature", shift_std=2.5).corrupted_df

    drift_scorer = DomainClassifierDriftScorer(epochs=20, random_state=42)

    auc_clean, drift_clean, _, _ = drift_scorer.score(ref_df, cur_clean)
    auc_shifted, drift_shifted, _, _ = drift_scorer.score(ref_df, cur_shifted)

    print(f"Clean Batch -> Domain Classifier ROC AUC: {auc_clean:.3f} | Drift Alert: {drift_clean}")
    print(f"Shifted Batch (+2.5 sigma) -> Domain Classifier ROC AUC: {auc_shifted:.3f} | Drift Alert: {drift_shifted}")

    return {
        "clean_auc": auc_clean,
        "clean_alert": drift_clean,
        "shifted_auc": auc_shifted,
        "shifted_alert": drift_shifted,
    }


def main():
    print("=" * 65)
    print("DataSentry Phase 3: PyTorch Deep Learning Anomaly & Drift Ablation")
    print("=" * 65)

    # 1. Anomaly detection ablation
    runner = AnomalyAblationRunner(random_state=42)
    print("\nRunning 10-trial Anomaly Detector Ablation across 4 models...")
    ablation_summary = runner.run_ablation(n_trials=10)

    print("\n--- Anomaly Detector Ablation Results ---")
    print(ablation_summary.round(4).to_string(index=False))

    # 2. Domain classifier drift evaluation
    drift_res = run_drift_ablation()

    # Save to outputs
    os.makedirs("outputs", exist_ok=True)
    ablation_summary.round(4).to_csv("outputs/ablation_results.csv", index=False)

    from datasentry.evaluation.benchmark import dataframe_to_markdown

    md_table = dataframe_to_markdown(ablation_summary.round(4))
    with open("outputs/ablation_results.md", "w", encoding="utf-8") as f:
        f.write("# Phase 3: Deep Learning Component Ablation Results\n\n")
        f.write("Ablation study comparing PyTorch Autoencoder against Isolation Forest, Mahalanobis Distance, ")
        f.write("and PCA Reconstruction Error on sensor telemetry with injected anomalies.\n\n")
        f.write(md_table)
        f.write("\n\n### Learned Domain Classifier Drift Metrics\n")
        f.write(f"- **Clean Negative Control ROC AUC:** {drift_res['clean_auc']:.3f} (Alert: {drift_res['clean_alert']})\n")
        f.write(f"- **Shifted Batch (+2.5 sigma) ROC AUC:** {drift_res['shifted_auc']:.3f} (Alert: {drift_res['shifted_alert']})\n")

    print("\nAblation artifacts saved to outputs/ablation_results.csv and outputs/ablation_results.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
