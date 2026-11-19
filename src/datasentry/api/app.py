"""
DataSentry FastAPI Service - Production REST API

Endpoints:
- GET  /health: System health, configuration, and security policy metadata
- POST /scan: Ingest reference & current records and return 3-axis reliability report
- POST /explain: Generate grounded watsonx Granite explanation with faithfulness audit
- POST /remediate: Autonomous, safety-gated pipeline remediation with HITL gates
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from datasentry.agents.guardrails import GuardrailEngine, SecurityPolicy
from datasentry.agents.remediation import RemediationAgent
from datasentry.api.dashboard import DASHBOARD_HTML
from datasentry.core.engine import DataSentryEngine
from datasentry.explainability.explainer import DataSentryExplainer, ExplanationResult
from datasentry.explainability.verifier import FaithfulnessVerifier
from datasentry.schemas import DataSentryReport

app = FastAPI(
    title="DataSentry API",
    description="Agentic, Safety-Gated Data-Reliability Engine for Enterprise ML Pipelines",
    version="0.1.0",
)

# Global shared guardrail engine
_policy = SecurityPolicy.from_yaml("policies/security_policy.yaml") if os.path.exists("policies/security_policy.yaml") else SecurityPolicy()
_guardrails = GuardrailEngine(policy=_policy)
_agent = RemediationAgent(guardrails=_guardrails, dry_run=True)
_explainer = DataSentryExplainer()
_verifier = FaithfulnessVerifier()


class ScanRequest(BaseModel):
    dataset_name: str = "production_feed"
    current_records: List[Dict[str, Any]]
    reference_records: Optional[List[Dict[str, Any]]] = None


class ExplainRequest(BaseModel):
    report: DataSentryReport
    verify_faithfulness: bool = True


class ExplainResponse(BaseModel):
    explanation: ExplanationResult
    faithfulness_score: float
    has_hallucinations: bool
    citation_precision: float
    citation_recall: float


class RemediateRequest(BaseModel):
    report: DataSentryReport
    role: str = "remediation_agent"
    human_approved: bool = False
    dry_run: bool = True


class RemediateResponse(BaseModel):
    action: str
    tool_called: str
    allowed: bool
    executed: bool
    reason: str
    risk_tier: str


@app.get("/", response_class=HTMLResponse, status_code=status.HTTP_200_OK)
def dashboard_view() -> str:
    """Interactive single-page web dashboard for live monitoring and demonstrations."""
    return DASHBOARD_HTML


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Health check and active security configuration."""
    return {
        "status": "healthy",
        "service": "datasentry",
        "version": "0.1.0",
        "dry_run_default": _policy.dry_run_default,
        "max_steps": _policy.max_execution_steps,
        "rate_limit_per_minute": _policy.rate_limit_per_minute,
        "active_roles": list(_policy.roles.keys()),
    }


@app.post("/scan", response_model=DataSentryReport, status_code=status.HTTP_200_OK)
def scan_endpoint(req: ScanRequest) -> DataSentryReport:
    """Run 3-axis quality validation, anomaly detection, and drift monitoring."""
    if not req.current_records:
        raise HTTPException(status_code=400, detail="current_records cannot be empty.")

    curr_df = pd.DataFrame(req.current_records)
    ref_df = pd.DataFrame(req.reference_records) if req.reference_records else None

    engine = DataSentryEngine()
    report = engine.scan(current_df=curr_df, reference_df=ref_df, dataset_name=req.dataset_name)
    return report


@app.post("/explain", response_model=ExplainResponse, status_code=status.HTTP_200_OK)
def explain_endpoint(req: ExplainRequest) -> ExplainResponse:
    """Generate plain-language diagnosis grounded in report numbers with citation auditing."""
    explanation = _explainer.explain(req.report)

    f_score = 1.0
    has_hallucination = False
    cit_prec = 1.0
    cit_rec = 1.0

    if req.verify_faithfulness:
        audit_res = _verifier.verify(req.report, explanation)
        f_score = audit_res.overall_faithfulness
        has_hallucination = not audit_res.is_faithful
        cit_prec = audit_res.citation_precision
        cit_rec = audit_res.citation_recall

    return ExplainResponse(
        explanation=explanation,
        faithfulness_score=f_score,
        has_hallucinations=has_hallucination,
        citation_precision=cit_prec,
        citation_recall=cit_rec,
    )


@app.post("/remediate", response_model=RemediateResponse, status_code=status.HTTP_200_OK)
def remediate_endpoint(req: RemediateRequest) -> RemediateResponse:
    """Execute safety-gated autonomous remediation on a DataSentry report."""
    agent = RemediationAgent(guardrails=_guardrails, role=req.role, dry_run=req.dry_run)
    res = agent.act(report=req.report, human_approved=req.human_approved)

    return RemediateResponse(
        action=res.action_selected.value,
        tool_called=res.tool_invoked,
        allowed=res.guardrail_decision.allowed,
        executed=res.executed,
        reason=res.guardrail_decision.reason,
        risk_tier=res.guardrail_decision.risk_tier.value,
    )
