"""
DataSentry CLI - Unified Command-Line Interface

Commands:
- scan: Inspect data quality, record anomalies, and dataset drift.
- explain: Generate grounded Granite explanation with citation auditing.
- agent: Trigger safety-gated autonomous remediation.
- eval: Execute baseline benchmarks, PyTorch ablations, or red-team evaluations.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import click
import pandas as pd

from datasentry.agents.guardrails import GuardrailEngine, SecurityPolicy
from datasentry.agents.remediation import RemediationAgent
from datasentry.core.engine import DataSentryEngine
from datasentry.explainability.explainer import DataSentryExplainer
from datasentry.explainability.verifier import FaithfulnessVerifier


@click.group()
@click.version_option("0.1.0", prog_name="datasentry")
def cli():
    """DataSentry: Agentic, Safety-Gated Data-Reliability Engine for ML Pipelines."""
    pass


@cli.command("scan")
@click.option("--current", "-c", required=True, type=click.Path(exists=True), help="Path to current batch CSV/parquet.")
@click.option("--reference", "-r", required=False, type=click.Path(exists=True), default=None, help="Path to reference baseline CSV/parquet.")
@click.option("--name", "-n", default="tabular_batch", help="Dataset identifier name.")
@click.option("--output", "-o", default=None, help="Path to save JSON report.")
def scan_command(current: str, reference: Optional[str], name: str, output: Optional[str]):
    """Run 3-axis quality validation, anomaly detection, and drift monitoring."""
    click.echo(f"[*] Scanning dataset '{name}'...")
    curr_df = pd.read_csv(current) if current.endswith(".csv") else pd.read_parquet(current)
    ref_df = None
    if reference:
        ref_df = pd.read_csv(reference) if reference.endswith(".csv") else pd.read_parquet(reference)

    engine = DataSentryEngine()
    report = engine.scan(current_df=curr_df, reference_df=ref_df, dataset_name=name)

    click.echo(f"[+] Scan Complete: Status = {report.overall_status.upper()}")
    click.echo(f"    - Quality issues: {len(report.quality.schema_violations)}")
    click.echo(f"    - Anomalies: {report.anomaly.anomaly_count} ({report.anomaly.anomaly_percentage:.1f}%)")
    drift_count = report.drift.drifted_features_count if report.drift else 0
    click.echo(f"    - Drifted features: {drift_count}")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        click.echo(f"[+] Saved structured report to {output}")
    else:
        click.echo(report.model_dump_json(indent=2))


@cli.command("explain")
@click.option("--report-path", "-r", required=True, type=click.Path(exists=True), help="Path to DataSentry JSON report.")
@click.option("--verify/--no-verify", default=True, help="Run faithfulness and hallucination verification.")
def explain_command(report_path: str, verify: bool):
    """Generate plain-language diagnosis grounded in report numbers with citation auditing."""
    click.echo(f"[*] Loading report from {report_path}...")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from datasentry.schemas import DataSentryReport
    report = DataSentryReport.model_validate(data)

    explainer = DataSentryExplainer()
    explanation = explainer.explain(report)

    click.echo("\n" + "=" * 60)
    click.echo("DATASENTRY ROOT-CAUSE DIAGNOSIS (watsonx.ai Granite)")
    click.echo("=" * 60)
    click.echo(f"\n{explanation.explanation}\n")
    click.echo(f"Cited Findings: {', '.join(explanation.cited_finding_ids)}")

    if verify:
        verifier = FaithfulnessVerifier()
        audit_res = verifier.verify(report, explanation)
        click.echo("\n--- FAITHFULNESS & GROUNDING AUDIT ---")
        click.echo(f"Faithfulness Score: {audit_res.overall_faithfulness * 100:.1f}%")
        click.echo(f"Hallucination Detected: {not audit_res.is_faithful}")
        click.echo(f"Citation Precision: {audit_res.citation_precision * 100:.1f}%")
        click.echo(f"Numeric Precision: {audit_res.numeric_faithfulness * 100:.1f}%")


@cli.command("agent")
@click.option("--report-path", "-r", required=True, type=click.Path(exists=True), help="Path to DataSentry JSON report.")
@click.option("--policy", "-p", default="policies/security_policy.yaml", help="Path to security policy YAML.")
@click.option("--role", default="remediation_agent", help="Agent execution role.")
@click.option("--approve/--no-approve", default=False, help="Human approval token for high-risk tools.")
@click.option("--dry-run/--live", default=True, help="Simulate execution in dry-run mode.")
def agent_command(report_path: str, policy: str, role: str, approve: bool, dry_run: bool):
    """Execute safety-gated autonomous remediation on a DataSentry report."""
    click.echo(f"[*] Initializing Remediation Agent [Role: {role}, DryRun: {dry_run}]...")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from datasentry.schemas import DataSentryReport
    report = DataSentryReport.model_validate(data)

    sec_policy = SecurityPolicy.from_yaml(policy) if os.path.exists(policy) else SecurityPolicy()
    guardrails = GuardrailEngine(policy=sec_policy)
    agent = RemediationAgent(guardrails=guardrails, role=role, dry_run=dry_run)

    result = agent.act(report=report, human_approved=approve)
    click.echo(f"[+] Action Decision: {result.action_selected.value.upper()}")
    click.echo(f"    Tool Selected: {result.tool_invoked}")
    click.echo(f"    Guardrail Allowed: {result.guardrail_decision.allowed}")
    click.echo(f"    Executed: {result.executed}")
    click.echo(f"    Reasoning: {result.guardrail_decision.reason}")


@cli.command("eval")
@click.argument("suite", type=click.Choice(["baseline", "ablation", "redteam", "all"]))
def eval_command(suite: str):
    """Run benchmark evaluation suites (baseline, ablation, redteam, or all)."""
    click.echo(f"[*] Running evaluation suite: {suite.upper()}...")
    if suite in ("baseline", "all"):
        click.echo("--- Running Phase 1 Baseline Benchmark ---")
        from datasentry.evaluation.benchmark import BaselineBenchmarkRunner
        runner = BaselineBenchmarkRunner()
        runner.run_benchmark(output_csv="outputs/benchmark_results.csv", output_md="outputs/benchmark_results.md")

    if suite in ("ablation", "all"):
        click.echo("--- Running Phase 3 PyTorch Anomaly & Drift Ablation ---")
        from datasentry.evaluation.ablation import run_ablation_benchmarks
        run_ablation_benchmarks()

    if suite in ("redteam", "all"):
        click.echo("--- Running Phase 5 Adversarial Red-Team Evaluation ---")
        from datasentry.evaluation.red_team import RedTeamRunner
        rt = RedTeamRunner()
        df = rt.run_benchmark()
        rt.generate_summary_tables(df, output_csv="outputs/redteam_results.csv", output_md="outputs/redteam_results.md")

    click.echo("[+] Evaluation complete. Results saved in outputs/")


if __name__ == "__main__":
    cli()
