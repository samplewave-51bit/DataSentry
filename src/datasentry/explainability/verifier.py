"""
DataSentry Explainability - Faithfulness & Hallucination Auditor

Automatically checks that every finding ID, column name, and numeric value
cited in the natural-language explanation strictly exists in the source report.
Measures entity-level faithfulness and hallucination rate.
"""

from __future__ import annotations

import re
from typing import List, Set

from pydantic import BaseModel, Field

from datasentry.explainability.explainer import ExplanationResult
from datasentry.schemas import DataSentryReport, Severity


class FaithfulnessReport(BaseModel):
    report_id: str
    total_findings_in_report: int
    cited_findings_count: int
    valid_citations_count: int
    invalid_citations: List[str] = Field(default_factory=list)
    citation_precision: float
    citation_recall: float
    mentioned_columns: List[str] = Field(default_factory=list)
    hallucinated_columns: List[str] = Field(default_factory=list)
    column_faithfulness: float
    mentioned_numbers_count: int
    hallucinated_numbers: List[float] = Field(default_factory=list)
    numeric_faithfulness: float
    overall_faithfulness: float
    hallucination_rate: float
    is_faithful: bool


class FaithfulnessVerifier:
    """Audits LLM explanations against structured DataSentry reports."""

    def __init__(self, numeric_tolerance: float = 0.05):
        self.numeric_tolerance = numeric_tolerance

    def verify(
        self, report: DataSentryReport, explanation_res: ExplanationResult
    ) -> FaithfulnessReport:
        text = explanation_res.explanation

        # 1. Audit Finding ID Citations
        report_finding_ids: Set[str] = {f.finding_id for f in report.all_findings}
        cited_ids = set(explanation_res.cited_finding_ids)

        valid_citations = cited_ids.intersection(report_finding_ids)
        invalid_citations = list(cited_ids - report_finding_ids)

        cit_precision = float(len(valid_citations) / len(cited_ids)) if cited_ids else 1.0
        # Citation recall on high/critical findings
        key_findings = {f.finding_id for f in report.all_findings if f.severity in (Severity.CRITICAL, Severity.HIGH)}
        cit_recall = (
            float(len(valid_citations.intersection(key_findings)) / len(key_findings))
            if key_findings
            else 1.0
        )

        # 2. Audit Column Mentions
        valid_columns = set(report.quality.column_metrics.keys())
        for f in report.all_findings:
            if f.column:
                valid_columns.add(f.column)

        # Ignore dataset_name, report_id, or common structural names
        ignored_names = {report.dataset_name, report.report_id, "dataset", "batch", "table"}

        # Match single-word tokens in single/double quotes or after 'column'
        quoted_tokens = re.findall(r"['`]([a-zA-Z0-9_]+)['`]", text)
        explicit_col_tokens = re.findall(r"column\s+['`]?([a-zA-Z0-9_]+)['`]?", text, re.IGNORECASE)
        candidate_cols = list(dict.fromkeys(quoted_tokens + explicit_col_tokens))

        mentioned_cols = [
            c for c in candidate_cols
            if c not in ignored_names and not c.startswith("FND-")
        ]

        hallucinated_cols = [col for col in mentioned_cols if col not in valid_columns]
        col_faithfulness = (
            1.0 - float(len(hallucinated_cols) / len(mentioned_cols))
            if mentioned_cols
            else 1.0
        )

        # 3. Audit Numeric Mentions
        ground_truth_numbers = self._extract_ground_truth_numbers(report)

        # Extract factual numbers from text, filtering out section indices (1., 2., 3.) and years
        number_tokens = re.findall(r"(?<![a-zA-Z0-9_])\d+(?:\.\d+)?%?", text)
        filtered_numbers: List[float] = []
        for tok in number_tokens:
            is_pct = tok.endswith("%")
            val_str = tok.rstrip("%")
            try:
                val = float(val_str)
                # Ignore section numbers (1, 2, 3) and obvious years
                if val in (1.0, 2.0, 3.0, 4.0) and not is_pct:
                    continue
                if 2020 <= val <= 2030:
                    continue
                filtered_numbers.append(val / 100.0 if is_pct else val)
            except ValueError:
                pass

        hallucinated_numbers = []
        for n_val in filtered_numbers:
            matched = any(
                self._is_close(n_val, gt) or self._is_close(n_val * 100.0, gt) or self._is_close(n_val / 100.0, gt)
                for gt in ground_truth_numbers
            )
            if not matched:
                hallucinated_numbers.append(n_val)

        num_faithfulness = (
            1.0 - float(len(hallucinated_numbers) / len(filtered_numbers))
            if filtered_numbers
            else 1.0
        )

        # 4. Overall Faithfulness & Hallucination Rate
        weights = [cit_precision, col_faithfulness, num_faithfulness]
        overall_faithfulness = float(sum(weights) / len(weights))
        hallucination_rate = 1.0 - overall_faithfulness

        is_faithful = bool(
            len(invalid_citations) == 0
            and len(hallucinated_cols) == 0
            and len(hallucinated_numbers) <= 1
            and overall_faithfulness >= 0.85
        )

        return FaithfulnessReport(
            report_id=report.report_id,
            total_findings_in_report=len(report.all_findings),
            cited_findings_count=len(cited_ids),
            valid_citations_count=len(valid_citations),
            invalid_citations=invalid_citations,
            citation_precision=round(cit_precision, 4),
            citation_recall=round(cit_recall, 4),
            mentioned_columns=mentioned_cols,
            hallucinated_columns=hallucinated_cols,
            column_faithfulness=round(col_faithfulness, 4),
            mentioned_numbers_count=len(filtered_numbers),
            hallucinated_numbers=[round(n, 4) for n in hallucinated_numbers],
            numeric_faithfulness=round(num_faithfulness, 4),
            overall_faithfulness=round(overall_faithfulness, 4),
            hallucination_rate=round(hallucination_rate, 4),
            is_faithful=is_faithful,
        )

    def _is_close(self, a: float, b: float) -> bool:
        if abs(a - b) < 1e-4:
            return True
        denom = max(abs(a), abs(b), 1e-4)
        return (abs(a - b) / denom) <= self.numeric_tolerance

    def _extract_ground_truth_numbers(self, report: DataSentryReport) -> List[float]:
        numbers: List[float] = [
            float(report.quality.total_rows),
            float(report.quality.total_columns),
            float(report.quality.duplicate_rows),
            float(report.quality.duplicate_percentage),
            float(report.anomaly.total_evaluated),
            float(report.anomaly.anomaly_count),
            float(report.anomaly.anomaly_percentage),
        ]
        if report.drift is not None:
            numbers.extend([
                float(report.drift.total_features),
                float(report.drift.drifted_features_count),
                float(report.drift.drift_share),
                float(report.drift.drift_threshold_share),
            ])
            for m in report.drift.metrics.values():
                numbers.append(float(m.statistic))
                numbers.append(float(m.threshold))
                if m.p_value is not None:
                    numbers.append(float(m.p_value))

        for f in report.all_findings:
            numbers.append(float(f.metric_value))
            numbers.append(float(f.threshold))

        return numbers
