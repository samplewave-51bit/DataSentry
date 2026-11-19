"""
Phase 4: Explanation Layer & Faithfulness Evaluation Runner

Generates grounded natural-language explanations for injected-fault reports
and measures entity-level hallucination rates and citation precision.
Outputs saved to outputs/explanation_faithfulness.csv and outputs/explanation_faithfulness.md.
"""

import os
from typing import Any, Dict, List

import pandas as pd

from datasentry.core.engine import DataSentryEngine
from datasentry.evaluation.datasets import (
    load_adult_census,
    load_healthcare_vitals,
    load_sensor_telemetry,
)
from datasentry.evaluation.fault_injection import FaultInjector
from datasentry.explainability.explainer import DataSentryExplainer
from datasentry.explainability.verifier import FaithfulnessVerifier


def run_faithfulness_benchmark() -> pd.DataFrame:
    print("=" * 65)
    print("Running Phase 4 Grounded Explanation & Faithfulness Benchmark")
    print("=" * 65)

    datasets = {
        "adult_census": load_adult_census(n_samples=500),
        "sensor_telemetry": load_sensor_telemetry(n_samples=500),
        "healthcare_vitals": load_healthcare_vitals(n_samples=500),
    }

    injector = FaultInjector(random_state=42)
    engine = DataSentryEngine()
    explainer = DataSentryExplainer()
    verifier = FaithfulnessVerifier(numeric_tolerance=0.05)

    eval_records: List[Dict[str, Any]] = []

    for ds_name, (clean_df, schema) in datasets.items():
        ref_df = clean_df.iloc[:250].copy()
        test_df = clean_df.iloc[250:].copy().reset_index(drop=True)

        target_col = "hours_per_week" if ds_name == "adult_census" else ("temperature" if ds_name == "sensor_telemetry" else "glucose")

        # 1. Clean control
        rep_clean = engine.scan(current_df=test_df, reference_df=ref_df, dataset_name=ds_name)
        exp_clean = explainer.explain(rep_clean)
        ver_clean = verifier.verify(rep_clean, exp_clean)
        eval_records.append({
            "dataset": ds_name,
            "scenario": "CLEAN_CONTROL",
            "findings_count": len(rep_clean.all_findings),
            "cited_count": ver_clean.cited_findings_count,
            "citation_precision": ver_clean.citation_precision,
            "citation_recall": ver_clean.citation_recall,
            "column_faithfulness": ver_clean.column_faithfulness,
            "numeric_faithfulness": ver_clean.numeric_faithfulness,
            "hallucination_rate": ver_clean.hallucination_rate,
            "is_faithful": ver_clean.is_faithful,
            "latency_ms": exp_clean.latency_ms,
        })

        # 2. Injected Missingness Spike
        f_missing = injector.inject_missingness_spike(test_df, column=target_col, missing_rate=0.40)
        rep_missing = engine.scan(current_df=f_missing.corrupted_df, reference_df=ref_df, dataset_name=ds_name)
        exp_missing = explainer.explain(rep_missing)
        ver_missing = verifier.verify(rep_missing, exp_missing)
        eval_records.append({
            "dataset": ds_name,
            "scenario": "MISSINGNESS_SPIKE",
            "findings_count": len(rep_missing.all_findings),
            "cited_count": ver_missing.cited_findings_count,
            "citation_precision": ver_missing.citation_precision,
            "citation_recall": ver_missing.citation_recall,
            "column_faithfulness": ver_missing.column_faithfulness,
            "numeric_faithfulness": ver_missing.numeric_faithfulness,
            "hallucination_rate": ver_missing.hallucination_rate,
            "is_faithful": ver_missing.is_faithful,
            "latency_ms": exp_missing.latency_ms,
        })

        # 3. Injected Covariate Shift
        f_shift = injector.inject_covariate_shift(test_df, column=target_col, shift_std=3.0)
        rep_shift = engine.scan(current_df=f_shift.corrupted_df, reference_df=ref_df, dataset_name=ds_name)
        exp_shift = explainer.explain(rep_shift)
        ver_shift = verifier.verify(rep_shift, exp_shift)
        eval_records.append({
            "dataset": ds_name,
            "scenario": "COVARIATE_SHIFT",
            "findings_count": len(rep_shift.all_findings),
            "cited_count": ver_shift.cited_findings_count,
            "citation_precision": ver_shift.citation_precision,
            "citation_recall": ver_shift.citation_recall,
            "column_faithfulness": ver_shift.column_faithfulness,
            "numeric_faithfulness": ver_shift.numeric_faithfulness,
            "hallucination_rate": ver_shift.hallucination_rate,
            "is_faithful": ver_shift.is_faithful,
            "latency_ms": exp_shift.latency_ms,
        })

        # 4. Injected Duplicate Rows
        f_dup = injector.inject_duplicate_rows(test_df, fraction=0.15)
        rep_dup = engine.scan(current_df=f_dup.corrupted_df, reference_df=ref_df, dataset_name=ds_name)
        exp_dup = explainer.explain(rep_dup)
        ver_dup = verifier.verify(rep_dup, exp_dup)
        eval_records.append({
            "dataset": ds_name,
            "scenario": "DUPLICATE_ROWS",
            "findings_count": len(rep_dup.all_findings),
            "cited_count": ver_dup.cited_findings_count,
            "citation_precision": ver_dup.citation_precision,
            "citation_recall": ver_dup.citation_recall,
            "column_faithfulness": ver_dup.column_faithfulness,
            "numeric_faithfulness": ver_dup.numeric_faithfulness,
            "hallucination_rate": ver_dup.hallucination_rate,
            "is_faithful": ver_dup.is_faithful,
            "latency_ms": exp_dup.latency_ms,
        })

    df_eval = pd.DataFrame(eval_records)
    return df_eval


def main():
    df_eval = run_faithfulness_benchmark()

    # Aggregate metrics
    summary = pd.DataFrame([{
        "Total Reports Evaluated": len(df_eval),
        "Mean Citation Precision": f"{df_eval['citation_precision'].mean():.2%}",
        "Mean Citation Recall": f"{df_eval['citation_recall'].mean():.2%}",
        "Mean Column Faithfulness": f"{df_eval['column_faithfulness'].mean():.2%}",
        "Mean Numeric Faithfulness": f"{df_eval['numeric_faithfulness'].mean():.2%}",
        "Overall Hallucination Rate": f"{df_eval['hallucination_rate'].mean():.2%}",
        "Faithful Explanations Ratio": f"{df_eval['is_faithful'].mean():.2%}",
        "Average Latency (ms)": f"{df_eval['latency_ms'].mean():.2f}",
    }])

    print("\n--- Faithfulness & Hallucination Summary ---")
    print(summary.to_string(index=False))
    print("\n--- Detailed Per-Scenario Evaluation ---")
    print(df_eval[["dataset", "scenario", "citation_precision", "numeric_faithfulness", "hallucination_rate"]].to_string(index=False))

    # Save outputs
    os.makedirs("outputs", exist_ok=True)
    df_eval.to_csv("outputs/explanation_faithfulness.csv", index=False)

    from datasentry.evaluation.benchmark import dataframe_to_markdown

    with open("outputs/explanation_faithfulness.md", "w", encoding="utf-8") as f:
        f.write("# Phase 4: Explanation Layer Faithfulness & Hallucination Results\n\n")
        f.write("Evaluation of Grounded LLM Explainer across injected faults on 3 tabular datasets.\n\n")
        f.write(dataframe_to_markdown(summary))
        f.write("\n\n### Detailed Evaluations by Scenario\n\n")
        f.write(dataframe_to_markdown(df_eval[["dataset", "scenario", "findings_count", "cited_count", "citation_precision", "numeric_faithfulness", "hallucination_rate", "is_faithful"]]))
        f.write("\n")

    print("\nOutputs saved to outputs/explanation_faithfulness.csv and outputs/explanation_faithfulness.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
