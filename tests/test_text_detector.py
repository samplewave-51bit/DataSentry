"""Unit tests for TextDriftDetector (Phase 2)."""

import pytest

from datasentry.detectors.text import TextDriftDetector, compute_rbf_mmd


@pytest.fixture
def clean_and_drifted_corpora():
    # Reference corpus: Healthcare / Clinical records
    ref_docs = [
        "Patient exhibits elevated blood glucose levels and symptoms of type 2 diabetes mellitus.",
        "Clinical diagnosis indicates hypertension, high systolic pressure, and cardiovascular strain.",
        "Prescribed metformin daily dosage alongside regular blood pressure monitoring and dietary adjustments.",
        "Patient medical history includes asthma, mild tachycardia, and routine annual physical examinations.",
        "Laboratory results show elevated cholesterol and LDL triglycerides requiring statin therapy.",
        "The intensive care physician observed stable vital signs with normalized oxygen saturation post-treatment.",
    ]

    # Similar current corpus (no drift)
    similar_cur_docs = [
        "Follow-up visit confirms stable glycemic control with prescribed metformin regimen.",
        "Cardiovascular examination showed controlled blood pressure and normal resting heart rate.",
        "Patient reports adherence to low-sodium diet with improved systolic blood pressure metrics.",
        "Routine laboratory blood tests confirmed normalized cholesterol levels post statin adjustment.",
        "Clinical evaluation reveals no acute cardiopulmonary distress in diabetes follow-up patient.",
    ]

    # Shifted current corpus: Enterprise Cloud Computing & Kubernetes
    drifted_cur_docs = [
        "The microservice container crashed due to out-of-memory OOM exceptions in the Kubernetes pod cluster.",
        "Configured ingress controller and load balancer to distribute traffic across distributed database nodes.",
        "Deployed CI/CD pipeline automated release on Google Cloud Platform with Terraform infrastructure state.",
        "Kubernetes horizontal pod autoscaler scaled out deployment replicas during severe network latency spike.",
        "Upgraded Apache Spark cluster drivers and optimized distributed memory shuffle partitions.",
    ]

    return ref_docs, similar_cur_docs, drifted_cur_docs


def test_rbf_mmd_computation():
    import numpy as np

    np.random.seed(42)
    # Identical distributions
    x = np.random.normal(0, 1, size=(50, 10))
    y = np.random.normal(0, 1, size=(50, 10))
    mmd_low = compute_rbf_mmd(x, y)

    # Shifted distribution
    z = np.random.normal(5, 1, size=(50, 10))
    mmd_high = compute_rbf_mmd(x, z)

    assert mmd_high > mmd_low


def test_text_detector_similar_corpus(clean_and_drifted_corpora):
    ref_docs, similar_docs, _ = clean_and_drifted_corpora
    detector = TextDriftDetector(auc_threshold=0.75, cosine_threshold=0.30)
    report = detector.detect(current_data=similar_docs, reference_data=ref_docs)

    assert report.total_reference_docs == len(ref_docs)
    assert report.total_current_docs == len(similar_docs)
    assert report.is_drift is False
    assert report.domain_classifier_auc < 0.75


def test_text_detector_detects_topic_drift(clean_and_drifted_corpora):
    ref_docs, _, drifted_docs = clean_and_drifted_corpora
    detector = TextDriftDetector(auc_threshold=0.65, cosine_threshold=0.15)
    report = detector.detect(current_data=drifted_docs, reference_data=ref_docs)

    assert report.is_drift is True
    # Domain classifier should easily discriminate healthcare vs cloud computing
    assert report.domain_classifier_auc >= 0.75
    assert report.centroid_cosine_distance > 0.20
    assert len(report.top_divergent_terms) > 0
    assert any(f.finding_id == "FND-TEXT-DRIFT-001" for f in report.findings)
