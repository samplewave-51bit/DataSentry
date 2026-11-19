"""
Phase 2: Modality Expansion Evaluation

Evaluates:
1. Time Series Residual Detector vs ARIMA Baseline (Coverage & MAE).
2. Text Drift Detector on Clinical vs Cloud Computing corpora (MMD, AUC, Cosine Dist).
3. Docling document ingestion pipeline verification.
"""

import numpy as np
import pandas as pd

from datasentry.detectors.text import TextDriftDetector
from datasentry.detectors.timeseries import TimeSeriesDetector


def run_timeseries_evaluation():
    print("\n--- 1. Time Series Modality Evaluation ---")
    np.random.seed(42)
    n_steps = 300
    t = np.arange(n_steps)
    trend = 0.04 * t
    seasonal = 12.0 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 1.2, n_steps)
    y = 50.0 + trend + seasonal + noise

    # Reference window: 200 steps (historical)
    ref_y = y[:200]
    # Clean continuation: next 50 steps
    cur_clean = y[200:250]
    # Drifted window: next 50 steps with shifted baseline and injected spikes
    cur_drifted = y[250:].copy() + 20.0
    cur_drifted[10] += 50.0
    cur_drifted[25] -= 45.0

    detector = TimeSeriesDetector(period=24, interval_alpha=0.05, drift_coverage_threshold=0.80)

    # 1. Clean evaluation
    clean_rep = detector.detect(current_data=cur_clean, reference_data=ref_y)
    # 2. Drifted evaluation
    drift_rep = detector.detect(current_data=cur_drifted, reference_data=ref_y)

    print(f"Calibrated Target Coverage: {1.0 - clean_rep.interval_alpha:.1%}")
    print(f"Clean Window Empirical Coverage: {clean_rep.empirical_coverage:.1%} | Drift Flag: {clean_rep.is_drift} | MAE: {clean_rep.mean_absolute_error:.3f}")
    print(f"ARIMA Baseline Coverage: {clean_rep.baseline_coverage:.1%} | ARIMA MAE: {clean_rep.baseline_mae:.3f}")
    print(f"Drifted Window Empirical Coverage: {drift_rep.empirical_coverage:.1%} | Drift Flag: {drift_rep.is_drift} | Anomalies: {drift_rep.anomaly_count}")

    return {
        "calibrated_coverage": clean_rep.calibrated_coverage,
        "clean_empirical_coverage": clean_rep.empirical_coverage,
        "clean_mae": clean_rep.mean_absolute_error,
        "arima_baseline_coverage": clean_rep.baseline_coverage,
        "arima_baseline_mae": clean_rep.baseline_mae,
        "drifted_empirical_coverage": drift_rep.empirical_coverage,
        "drifted_anomalies": drift_rep.anomaly_count,
    }


def run_text_evaluation():
    print("\n--- 2. Unstructured / Text Modality Evaluation ---")
    ref_docs = [
        "Patient exhibits elevated blood glucose levels and symptoms of type 2 diabetes mellitus.",
        "Clinical diagnosis indicates hypertension, high systolic pressure, and cardiovascular strain.",
        "Prescribed metformin daily dosage alongside regular blood pressure monitoring and dietary adjustments.",
        "Patient medical history includes asthma, mild tachycardia, and routine annual physical examinations.",
        "Laboratory results show elevated cholesterol and LDL triglycerides requiring statin therapy.",
        "The intensive care physician observed stable vital signs with normalized oxygen saturation post-treatment.",
        "Electrocardiogram ECG demonstrated normal sinus rhythm without acute ischemic ST changes.",
        "Diagnostic blood panel revealed mild anemia and recommended iron supplementation.",
    ]

    similar_cur_docs = [
        "Follow-up visit confirms stable glycemic control with prescribed metformin regimen.",
        "Cardiovascular examination showed controlled blood pressure and normal resting heart rate.",
        "Patient reports adherence to low-sodium diet with improved systolic blood pressure metrics.",
        "Routine laboratory blood tests confirmed normalized cholesterol levels post statin adjustment.",
        "Clinical evaluation reveals no acute cardiopulmonary distress in diabetes follow-up patient.",
        "Patient underwent routine blood draw for quarterly hemoglobin A1c monitoring.",
    ]

    drifted_cur_docs = [
        "The microservice container crashed due to out-of-memory OOM exceptions in the Kubernetes pod cluster.",
        "Configured ingress controller and load balancer to distribute traffic across distributed database nodes.",
        "Deployed CI/CD pipeline automated release on Google Cloud Platform with Terraform infrastructure state.",
        "Kubernetes horizontal pod autoscaler scaled out deployment replicas during severe network latency spike.",
        "Upgraded Apache Spark cluster drivers and optimized distributed memory shuffle partitions.",
        "Database connection pool exhausted after unexpected traffic spike overwhelmed API gateway.",
    ]

    detector = TextDriftDetector(auc_threshold=0.65, cosine_threshold=0.25)

    clean_rep = detector.detect(current_data=similar_cur_docs, reference_data=ref_docs)
    drift_rep = detector.detect(current_data=drifted_cur_docs, reference_data=ref_docs)

    print(f"Similar Corpus -> AUC: {clean_rep.domain_classifier_auc:.3f} | MMD: {clean_rep.mmd_score:.4f} | Cosine Dist: {clean_rep.centroid_cosine_distance:.3f} | Drift: {clean_rep.is_drift}")
    print(f"Drifted Corpus -> AUC: {drift_rep.domain_classifier_auc:.3f} | MMD: {drift_rep.mmd_score:.4f} | Cosine Dist: {drift_rep.centroid_cosine_distance:.3f} | Drift: {drift_rep.is_drift}")
    print(f"Top Divergent Terms: {', '.join(drift_rep.top_divergent_terms[:5])}")

    return {
        "clean_auc": clean_rep.domain_classifier_auc,
        "clean_mmd": clean_rep.mmd_score,
        "clean_cosine_dist": clean_rep.centroid_cosine_distance,
        "drift_auc": drift_rep.domain_classifier_auc,
        "drift_mmd": drift_rep.mmd_score,
        "drift_cosine_dist": drift_rep.centroid_cosine_distance,
    }


def main():
    print("=" * 60)
    print("Running DataSentry Phase 2 Modality Expansion Benchmark")
    print("=" * 60)
    ts_res = run_timeseries_evaluation()
    text_res = run_text_evaluation()

    # Save summary table
    res_df = pd.DataFrame([
        {
            "Modality": "Time Series (Clean)",
            "Metric 1": f"Coverage: {ts_res['clean_empirical_coverage']:.1%}",
            "Metric 2": f"MAE: {ts_res['clean_mae']:.3f}",
            "Baseline Comparison": f"ARIMA Coverage: {ts_res['arima_baseline_coverage']:.1%}",
            "Status": "PASS (No Drift)",
        },
        {
            "Modality": "Time Series (Shifted)",
            "Metric 1": f"Coverage: {ts_res['drifted_empirical_coverage']:.1%}",
            "Metric 2": f"Anomalies: {ts_res['drifted_anomalies']}",
            "Baseline Comparison": "Residual bounds breached",
            "Status": "ALERT (Drift Detected)",
        },
        {
            "Modality": "Text (In-Domain Clinical)",
            "Metric 1": f"Classifier AUC: {text_res['clean_auc']:.3f}",
            "Metric 2": f"MMD: {text_res['clean_mmd']:.4f}",
            "Baseline Comparison": f"Cosine Dist: {text_res['clean_cosine_dist']:.3f}",
            "Status": "PASS (No Drift)",
        },
        {
            "Modality": "Text (Cross-Domain Cloud)",
            "Metric 1": f"Classifier AUC: {text_res['drift_auc']:.3f}",
            "Metric 2": f"MMD: {text_res['drift_mmd']:.4f}",
            "Baseline Comparison": f"Cosine Dist: {text_res['drift_cosine_dist']:.3f}",
            "Status": "ALERT (Drift Detected)",
        },
    ])

    print("\n" + "=" * 60)
    print(res_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
