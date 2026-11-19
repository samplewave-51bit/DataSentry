"""Unit tests for RemediationAgent (Phase 5)."""

import pandas as pd
import pytest

from datasentry.agents.remediation import RemediationAction, RemediationAgent
from datasentry.core.engine import DataSentryEngine


@pytest.fixture
def clean_report():
    df = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": [10.0, 20.0, 30.0]})
    engine = DataSentryEngine()
    return engine.scan(df, df, dataset_name="clean_pipeline")


@pytest.fixture
def corrupt_schema_report():
    ref_df = pd.DataFrame({"col_1": [1.0, 2.0], "col_2": [10.0, 20.0]})
    # Dropped mandatory column
    bad_df = pd.DataFrame({"col_1": [1.0, 2.0]})
    engine = DataSentryEngine(schema={"col_1": "numeric", "col_2": "numeric"})
    return engine.scan(bad_df, ref_df, dataset_name="bad_schema_pipeline")


def test_agent_passes_clean_batch(clean_report):
    agent = RemediationAgent(dry_run=True)
    res = agent.act(clean_report)
    assert res.action_selected == RemediationAction.PASS_BATCH
    assert res.tool_invoked == "pass_batch"
    assert res.executed is True
    assert res.dry_run is True


def test_agent_requests_schema_fix_on_schema_mismatch(corrupt_schema_report):
    agent = RemediationAgent(dry_run=True)
    res = agent.act(corrupt_schema_report)
    assert res.action_selected == RemediationAction.REQUEST_SCHEMA_FIX
    assert res.tool_invoked == "request_schema_fix"
    assert res.executed is True


def test_agent_holds_retraining_for_human_approval():
    # Simulate report with dataset drift
    ref_df = pd.DataFrame({"val": [1.0] * 50 + [2.0] * 50})
    cur_df = pd.DataFrame({"val": [100.0] * 50 + [200.0] * 50})  # Heavy drift
    engine = DataSentryEngine()
    drift_report = engine.scan(cur_df, ref_df)

    agent = RemediationAgent(dry_run=True)

    # Without approval: held by guardrail
    res_unapproved = agent.act(drift_report, human_approved=False)
    assert res_unapproved.action_selected == RemediationAction.TRIGGER_RETRAINING
    assert res_unapproved.executed is False
    assert res_unapproved.guardrail_decision.requires_human_approval is True

    # With approval: executed
    res_approved = agent.act(drift_report, human_approved=True)
    assert res_approved.action_selected == RemediationAction.TRIGGER_RETRAINING
    assert res_approved.executed is True
