"""
DataSentry Explainability - Grounded Report Explainer

Translates structured DataSentry reports into natural-language diagnoses
with mandatory Finding ID citations and strict numeric grounding.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional

from pydantic import BaseModel, Field

from datasentry.explainability.client import LLMClient, get_llm_client
from datasentry.schemas import DataSentryReport


class ExplanationResult(BaseModel):
    report_id: str
    dataset_name: str
    overall_status: str
    explanation: str
    cited_finding_ids: List[str] = Field(default_factory=list)
    referenced_columns: List[str] = Field(default_factory=list)
    referenced_numbers: List[float] = Field(default_factory=list)
    latency_ms: float = 0.0


class DataSentryExplainer:
    """Grounded Explainer translating DataSentry reports into plain language."""

    SYSTEM_PROMPT = (
        "You are DataSentry Explainer, an enterprise AI assistant for ML pipeline reliability. "
        "Your task is to convert the provided structured diagnostic report into an executive diagnosis.\n"
        "STRICT GROUNDING RULES:\n"
        "1. Every factual statement or diagnostic claim MUST cite its exact Finding ID in brackets (e.g. [FND-QUAL-001]).\n"
        "2. Reference ONLY the exact numbers, percentages, thresholds, and column names present in the report.\n"
        "3. Never speculate or hallucinate external causes not substantiated by the findings.\n"
        "4. Structure your response into: 1. Executive Summary, 2. Root Cause Analysis, 3. Recommended Remediation Plan."
    )

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_llm_client()

    def explain(self, report: DataSentryReport) -> ExplanationResult:
        """Generates grounded explanation for a DataSentryReport."""
        prompt = self._build_prompt(report)

        t0 = time.perf_counter()
        explanation_text = self.client.generate(prompt=prompt, system_prompt=self.SYSTEM_PROMPT)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Extract citations and referenced entities
        cited_ids = re.findall(r"\[(FND-[A-Z0-9-]+)\]", explanation_text)
        cited_ids_unique = list(dict.fromkeys(cited_ids))

        # Extract potential numbers
        raw_numbers = re.findall(r"[-+]?\d*\.?\d+", explanation_text)
        numbers = []
        for n_str in raw_numbers:
            try:
                numbers.append(float(n_str))
            except ValueError:
                pass

        # Extract column mentions
        report_columns = list(report.quality.column_metrics.keys())
        referenced_cols = [c for c in report_columns if c in explanation_text]

        return ExplanationResult(
            report_id=report.report_id,
            dataset_name=report.dataset_name,
            overall_status=report.overall_status,
            explanation=explanation_text,
            cited_finding_ids=cited_ids_unique,
            referenced_columns=referenced_cols,
            referenced_numbers=numbers,
            latency_ms=latency_ms,
        )

    def _build_prompt(self, report: DataSentryReport) -> str:
        lines = [
            f"Dataset: {report.dataset_name}",
            f"Report ID: {report.report_id}",
            f"Overall Gate Status: {report.overall_status}",
            f"Total Evaluated Rows: {report.quality.total_rows}",
            f"Total Columns: {report.quality.total_columns}",
            f"Duplicate Rows: {report.quality.duplicate_rows} ({report.quality.duplicate_percentage:.2%})",
            f"Record Anomalies Flagged: {report.anomaly.anomaly_count} ({report.anomaly.anomaly_percentage:.2%})",
        ]

        if report.drift is not None:
            lines.append(
                f"Dataset Drift Flag: {report.drift.dataset_drift} "
                f"({report.drift.drifted_features_count}/{report.drift.total_features} features drifted, "
                f"drift_share={report.drift.drift_share:.1%})"
            )

        if report.timeseries is not None:
            lines.append(
                f"Time Series Drift Flag: {report.timeseries.is_drift} "
                f"(empirical_coverage={report.timeseries.empirical_coverage:.1%}, "
                f"anomalies={report.timeseries.anomaly_count})"
            )

        if report.text is not None:
            lines.append(
                f"Text Drift Flag: {report.text.is_drift} "
                f"(classifier_auc={report.text.domain_classifier_auc:.3f}, "
                f"mmd={report.text.mmd_score:.4f}, cosine_dist={report.text.centroid_cosine_distance:.3f})"
            )

        lines.append("\nDiagnostic Findings List:")
        if not report.all_findings:
            lines.append("- (No findings; dataset passed all checks).")
        else:
            for f in report.all_findings:
                col_str = f" | Column: {f.column}" if f.column else ""
                lines.append(
                    f"- {f.finding_id}: [{f.severity.value}] Metric={f.metric_name}={f.metric_value:.4f} "
                    f"(Threshold={f.threshold:.4f}){col_str} | {f.description}"
                )

        return "\n".join(lines)
