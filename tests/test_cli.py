"""
Tests for DataSentry CLI commands.
"""

import pandas as pd
from click.testing import CliRunner

from datasentry.cli import cli


def test_cli_help():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "DataSentry" in res.output
    assert "scan" in res.output
    assert "explain" in res.output
    assert "agent" in res.output
    assert "eval" in res.output


def test_cli_scan_and_explain(tmp_path):
    runner = CliRunner()

    # Generate dummy batch
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "income": [50000, 60000, 75000, 90000],
    })
    curr_csv = tmp_path / "curr.csv"
    df.to_csv(curr_csv, index=False)
    report_json = tmp_path / "report.json"

    # Test scan
    res_scan = runner.invoke(cli, ["scan", "-c", str(curr_csv), "-o", str(report_json)])
    assert res_scan.exit_code == 0
    assert "Scan Complete" in res_scan.output
    assert report_json.exists()

    # Test explain
    res_explain = runner.invoke(cli, ["explain", "-r", str(report_json)])
    assert res_explain.exit_code == 0
    assert "DATASENTRY ROOT-CAUSE DIAGNOSIS" in res_explain.output
    assert "FAITHFULNESS & GROUNDING AUDIT" in res_explain.output

    # Test agent
    res_agent = runner.invoke(cli, ["agent", "-r", str(report_json)])
    assert res_agent.exit_code == 0
    assert "Action Decision:" in res_agent.output
