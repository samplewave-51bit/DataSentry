"""Unit tests for Red-Team Runner (Phase 5)."""

from datasentry.evaluation.red_team import RedTeamRunner


def test_red_team_benchmark_execution():
    runner = RedTeamRunner()
    assert len(runner.scenarios) >= 30

    df_results = runner.run_benchmark()
    assert len(df_results) == len(runner.scenarios)

    # In red-team benchmark, guardrails should successfully block 100% of malicious attempts
    blocked_count = int(df_results["guardrail_blocked"].sum())
    assert blocked_count == len(df_results)
