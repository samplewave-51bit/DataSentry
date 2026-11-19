"""
DataSentry Agents - Multi-Step Remediation Agent

Consumes DataSentry diagnostic reports, selects remediation actions,
and executes them through safety guardrails in dry-run or live mode.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from datasentry.agents.guardrails import GuardrailDecision, GuardrailEngine, InputSanitizer
from datasentry.schemas import DataSentryReport, Severity


class RemediationAction(str, Enum):
    PASS_BATCH = "pass_batch"
    QUARANTINE_BATCH = "quarantine_batch"
    REQUEST_SCHEMA_FIX = "request_schema_fix"
    TRIGGER_RETRAINING = "trigger_retraining"
    ESCALATE_TO_HUMAN = "escalate_to_human"


class RemediationResult(BaseModel):
    report_id: str
    dataset_name: str
    action_selected: RemediationAction
    tool_invoked: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    executed: bool
    dry_run: bool
    guardrail_decision: GuardrailDecision
    step_count: int
    rationale: str
    latency_ms: float = 0.0


class RemediationAgent:
    """Autonomous agent selecting and executing safety-gated remediation actions."""

    def __init__(
        self,
        guardrails: Optional[GuardrailEngine] = None,
        dry_run: bool = True,
        role: str = "remediation_agent",
    ):
        self.guardrails = guardrails or GuardrailEngine()
        self.dry_run = dry_run
        self.role = role

    def act(
        self,
        report: DataSentryReport,
        human_approved: bool = False,
        adversarial_override_tool: Optional[str] = None,
        adversarial_payload: Optional[Dict[str, Any]] = None,
        step_count: int = 1,
        role: Optional[str] = None,
    ) -> RemediationResult:
        """Evaluates report, decides action, passes through guardrails, and executes."""
        t0 = time.perf_counter()

        # Check for adversarial injection attempt or override
        if adversarial_override_tool:
            tool_to_call = adversarial_override_tool
            args = adversarial_payload or {}
            action = RemediationAction.ESCALATE_TO_HUMAN
            rationale = "Direct tool dispatch (adversarial/red-team test mode)."
        else:
            action, tool_to_call, args, rationale = self._decide_action(report)

        # Sanitize report description or argument text
        san_res = InputSanitizer.sanitize(rationale)
        if not san_res.is_safe:
            rationale = san_res.sanitized_text

        # Validate through Guardrail Layer
        decision = self.guardrails.validate_action(
            tool_name=tool_to_call,
            arguments=args,
            agent_role=role or self.role,
            human_approved=human_approved,
            step_count=step_count,
        )

        executed = False
        if decision.allowed:
            # Execute tool in dry-run or live
            executed = True
            if self.dry_run:
                self.guardrails.logger.log({
                    "action": "DRY_RUN_EXECUTE",
                    "tool": tool_to_call,
                    "arguments": args,
                    "status": "SIMULATED_SUCCESS",
                })
            else:
                self._dispatch_live_tool(tool_to_call, args)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return RemediationResult(
            report_id=report.report_id,
            dataset_name=report.dataset_name,
            action_selected=action,
            tool_invoked=tool_to_call,
            arguments=args,
            executed=executed,
            dry_run=self.dry_run,
            guardrail_decision=decision,
            step_count=step_count,
            rationale=rationale,
            latency_ms=latency_ms,
        )

    def _decide_action(
        self, report: DataSentryReport
    ) -> tuple[RemediationAction, str, Dict[str, Any], str]:
        """Deterministic policy decision based on diagnostic triad."""
        # 1. Clean batch -> Pass
        if report.overall_status == "PASS":
            return (
                RemediationAction.PASS_BATCH,
                "pass_batch",
                {"dataset": report.dataset_name, "report_id": report.report_id},
                "Batch certified clean across quality, anomaly, and drift axes.",
            )

        # 2. Schema violations -> Request schema fix
        if not report.quality.schema_matched:
            return (
                RemediationAction.REQUEST_SCHEMA_FIX,
                "request_schema_fix",
                {
                    "dataset": report.dataset_name,
                    "violations": report.quality.schema_violations,
                },
                f"Schema violations detected: {', '.join(report.quality.schema_violations[:2])}.",
            )

        # 3. Distribution drift -> Retraining trigger (HIGH-RISK)
        is_tabular_drift = report.drift is not None and report.drift.dataset_drift
        is_ts_drift = report.timeseries is not None and report.timeseries.is_drift
        is_text_drift = report.text is not None and report.text.is_drift

        if is_tabular_drift or is_ts_drift or is_text_drift:
            drift_reasons = []
            if is_tabular_drift:
                drift_reasons.append(f"tabular drift share={report.drift.drift_share:.1%}")
            if is_ts_drift:
                drift_reasons.append(f"time-series coverage={report.timeseries.empirical_coverage:.1%}")
            if is_text_drift:
                drift_reasons.append(f"text AUC={report.text.domain_classifier_auc:.3f}")

            return (
                RemediationAction.TRIGGER_RETRAINING,
                "trigger_retraining",
                {"dataset": report.dataset_name, "drift_summary": "; ".join(drift_reasons)},
                f"Population distribution shift requires model retraining: {'; '.join(drift_reasons)}.",
            )

        # 4. Critical quality errors or high duplicates -> Quarantine batch
        has_critical = any(f.severity == Severity.CRITICAL for f in report.all_findings)
        high_nulls = any(f.metric_name == "null_ratio" and f.severity == Severity.HIGH for f in report.all_findings)
        high_dups = report.quality.duplicate_percentage > 0.05

        if has_critical or high_nulls or high_dups:
            return (
                RemediationAction.QUARANTINE_BATCH,
                "quarantine_batch",
                {
                    "dataset": report.dataset_name,
                    "reason": "Severe data quality degradation or duplication.",
                },
                "Quarantining batch to prevent model corruption.",
            )

        # 5. Fallback -> Escalate to human
        return (
            RemediationAction.ESCALATE_TO_HUMAN,
            "notify_team",
            {"dataset": report.dataset_name, "status": report.overall_status},
            "Ambiguous findings require human auditor review.",
        )

    def _dispatch_live_tool(self, tool_name: str, args: Dict[str, Any]):
        """Executes live enterprise tool."""
        # Simulated live dispatch with audit log
        self.guardrails.logger.log({
            "action": "LIVE_EXECUTE",
            "tool": tool_name,
            "arguments": args,
            "status": "LIVE_COMPLETED",
        })
