"""Unit tests for FaithfulnessVerifier and Hallucination Auditor (Phase 4)."""

import pandas as pd
import pytest

from datasentry.core.engine import DataSentryEngine
from datasentry.explainability.explainer import DataSentryExplainer, ExplanationResult
from datasentry.explainability.verifier import FaithfulnessVerifier


@pytest.fixture
def sample_report_and_explainer():
    df = pd.DataFrame({"col_a": [1.0, 2.0, 3.0, 4.0], "col_b": [10.0, 20.0, 30.0, 40.0]})
    bad_df = pd.DataFrame({"col_a": [1.0, None, None, 4.0], "col_b": [10.0, 20.0, 30.0, 40.0]})

    engine = DataSentryEngine(schema={"col_a": "numeric", "col_b": "numeric"})
    report = engine.scan(bad_df, df, dataset_name="audit_test")
    explainer = DataSentryExplainer()
    return report, explainer


def test_verifier_on_faithful_explanation(sample_report_and_explainer):
    report, explainer = sample_report_and_explainer
    exp_res = explainer.explain(report)

    verifier = FaithfulnessVerifier()
    faith_rep = verifier.verify(report, exp_res)

    assert faith_rep.citation_precision == 1.0
    assert len(faith_rep.invalid_citations) == 0
    assert len(faith_rep.hallucinated_columns) == 0
    assert faith_rep.hallucination_rate <= 0.15
    assert faith_rep.is_faithful is True


def test_verifier_catches_hallucinations(sample_report_and_explainer):
    report, _ = sample_report_and_explainer

    # Craft an intentionally hallucinated explanation with fake finding ID, fake column, fake numbers
    hallucinated_text = (
        "Executive Summary: Dataset corrupted with 9999.88 null values. "
        "Root Cause: [FND-FAKE-999] indicates column 'unrelated_social_security_num' had 88.55% failure."
    )
    fake_exp_res = ExplanationResult(
        report_id=report.report_id,
        dataset_name=report.dataset_name,
        overall_status="FAIL",
        explanation=hallucinated_text,
        cited_finding_ids=["FND-FAKE-999"],
        referenced_columns=["unrelated_social_security_num"],
        referenced_numbers=[9999.88, 88.55],
    )

    verifier = FaithfulnessVerifier()
    faith_rep = verifier.verify(report, fake_exp_res)

    # Auditor must catch the invalid citation
    assert "FND-FAKE-999" in faith_rep.invalid_citations
    assert faith_rep.citation_precision == 0.0

    # Auditor must catch the hallucinated column
    assert "unrelated_social_security_num" in faith_rep.hallucinated_columns
    assert faith_rep.column_faithfulness == 0.0

    # Auditor must catch the fake numbers
    assert len(faith_rep.hallucinated_numbers) > 0
    assert faith_rep.hallucination_rate > 0.50
    assert faith_rep.is_faithful is False
