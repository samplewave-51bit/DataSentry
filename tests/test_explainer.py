"""Unit tests for DataSentryExplainer (Phase 4)."""

import pandas as pd
import pytest

from datasentry.core.engine import DataSentryEngine
from datasentry.explainability.client import MockGraniteClient
from datasentry.explainability.explainer import DataSentryExplainer


@pytest.fixture
def sample_report():
    ref_df = pd.DataFrame({"age": [25, 30, 45, 50], "salary": [50000, 60000, 80000, 90000]})
    # Injected nulls
    cur_df = pd.DataFrame({"age": [25, None, None, 50], "salary": [50000, 60000, 80000, 90000]})

    engine = DataSentryEngine(schema={"age": "numeric", "salary": "numeric"})
    return engine.scan(current_df=cur_df, reference_df=ref_df, dataset_name="payroll_data")


def test_mock_granite_generation(sample_report):
    client = MockGraniteClient()
    explainer = DataSentryExplainer(client=client)

    result = explainer.explain(sample_report)

    assert result.dataset_name == "payroll_data"
    assert "Executive Summary" in result.explanation
    assert "Root Cause Analysis" in result.explanation
    assert "Remediation Plan" in result.explanation
    assert len(result.cited_finding_ids) > 0
    # Every cited ID should start with FND-
    for fid in result.cited_finding_ids:
        assert fid.startswith("FND-")


def test_explainer_clean_dataset():
    clean_df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    engine = DataSentryEngine()
    clean_rep = engine.scan(clean_df, clean_df)

    explainer = DataSentryExplainer()
    result = explainer.explain(clean_rep)

    assert "certified clean" in result.explanation or "Executive Summary" in result.explanation
