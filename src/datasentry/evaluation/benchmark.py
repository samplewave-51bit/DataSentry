"""
DataSentry Evaluation - Benchmark Runner

Executes standardized benchmark evaluating DataSentry against:
- Naive KS-test
- Great Expectations
- Evidently

Across 3 public datasets with 10 fault scenarios and clean controls.
Measures: Precision, Recall, F1, False Alarm Rate, and Latency (ms).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from datasentry.evaluation.baselines import (
    BaseBenchmarkDetector,
    DataSentryEngineBaseline,
    EvidentlyBaseline,
    GreatExpectationsBaseline,
    NaiveKSBaseline,
)
from datasentry.evaluation.datasets import (
    load_adult_census,
    load_healthcare_vitals,
    load_sensor_telemetry,
)
from datasentry.evaluation.fault_injection import FaultInjector, FaultType


class BenchmarkRunner:
    """Orchestrates comprehensive fault injection benchmarking."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.injector = FaultInjector(random_state=random_state)
        self.detectors: List[BaseBenchmarkDetector] = [
            DataSentryEngineBaseline(),
            EvidentlyBaseline(),
            GreatExpectationsBaseline(),
            NaiveKSBaseline(),
        ]

    def run_benchmark(self, n_runs_per_fault: int = 5) -> pd.DataFrame:
        """Run full evaluation suite across all datasets and fault types."""
        datasets = {
            "adult_census": load_adult_census(),
            "sensor_telemetry": load_sensor_telemetry(),
            "healthcare_vitals": load_healthcare_vitals(),
        }

        records: List[Dict[str, Any]] = []

        for ds_name, (clean_df, schema) in datasets.items():
            # Split into reference and base test set
            split_idx = len(clean_df) // 2
            ref_df = clean_df.iloc[:split_idx].copy()
            base_test_df = clean_df.iloc[split_idx:].copy()

            # 1. Negative controls: Clean batches
            for run_i in range(n_runs_per_fault):
                # Clean sample without artificial duplicates
                sample_n = min(len(base_test_df), 200)
                sample_clean = base_test_df.sample(n=sample_n, replace=False, random_state=self.random_state + run_i)
                for det in self.detectors:
                    res = det.run(current_df=sample_clean, reference_df=ref_df, schema=schema)
                    records.append({
                        "dataset": ds_name,
                        "fault_type": "CLEAN_CONTROL",
                        "is_fault": False,
                        "detector": det.name,
                        "detected": res["detected_fault"],
                        "latency_ms": res["latency_ms"],
                    })

            # 2. Injected Faults
            fault_scenarios = self._generate_fault_scenarios(base_test_df, ds_name)
            for fault_type, corrupted_df in fault_scenarios:
                for det in self.detectors:
                    res = det.run(current_df=corrupted_df, reference_df=ref_df, schema=schema)
                    records.append({
                        "dataset": ds_name,
                        "fault_type": fault_type.value,
                        "is_fault": True,
                        "detector": det.name,
                        "detected": res["detected_fault"],
                        "latency_ms": res["latency_ms"],
                    })

        raw_df = pd.DataFrame(records)
        return self._compute_metrics(raw_df)

    def _generate_fault_scenarios(
        self, base_df: pd.DataFrame, ds_name: str
    ) -> List[tuple[FaultType, pd.DataFrame]]:
        scenarios = []

        # Determine target columns based on dataset
        if ds_name == "adult_census":
            num_col = "hours_per_week"
            cat_col = "income"
            col_a, col_b = "education_num", "hours_per_week"
        elif ds_name == "sensor_telemetry":
            num_col = "temperature"
            cat_col = "status"
            col_a, col_b = "temperature", "rotational_speed"
        else:
            num_col = "glucose"
            cat_col = "outcome"
            col_a, col_b = "glucose", "insulin"

        # 1. Schema Drop
        scenarios.append((FaultType.SCHEMA_DROP, self.injector.inject_schema_drop(base_df, num_col).corrupted_df))
        # 2. Schema Rename
        scenarios.append((FaultType.SCHEMA_RENAME, self.injector.inject_schema_rename(base_df, num_col).corrupted_df))
        # 3. Schema Retype
        scenarios.append((FaultType.SCHEMA_RETYPE, self.injector.inject_schema_retype(base_df, num_col).corrupted_df))
        # 4. Missingness Spike
        scenarios.append((FaultType.MISSINGNESS_SPIKE, self.injector.inject_missingness_spike(base_df, num_col, missing_rate=0.45).corrupted_df))
        # 5. Unit Error
        scenarios.append((FaultType.UNIT_ERROR, self.injector.inject_unit_error(base_df, num_col).corrupted_df))
        # 6. Out of Range
        scenarios.append((FaultType.OUT_OF_RANGE, self.injector.inject_out_of_range(base_df, num_col, fraction=0.08).corrupted_df))
        # 7. Duplicate Rows
        scenarios.append((FaultType.DUPLICATE_ROWS, self.injector.inject_duplicate_rows(base_df, fraction=0.15).corrupted_df))
        # 8. Covariate Shift
        scenarios.append((FaultType.COVARIATE_SHIFT, self.injector.inject_covariate_shift(base_df, num_col, shift_std=3.0).corrupted_df))
        # 9. Label Shift
        scenarios.append((FaultType.LABEL_SHIFT, self.injector.inject_label_shift(base_df, cat_col).corrupted_df))
        # 10. Correlated Feature Breakage
        scenarios.append((FaultType.CORRELATED_FEATURE_BREAKAGE, self.injector.inject_correlated_feature_breakage(base_df, col_a, col_b).corrupted_df))

        return scenarios

    def _compute_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for detector, d_group in df.groupby("detector"):
            fault_cases = d_group[d_group["is_fault"]]
            clean_cases = d_group[~d_group["is_fault"]]

            tp = int(fault_cases["detected"].sum())
            fn = int((~fault_cases["detected"]).sum())
            fp = int(clean_cases["detected"].sum())
            tn = int((~clean_cases["detected"]).sum())

            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            avg_latency = float(d_group["latency_ms"].mean())

            results.append({
                "detector": detector,
                "trials": len(d_group),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "false_alarm_rate": round(far, 4),
                "avg_latency_ms": round(avg_latency, 2),
            })

        res_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
        return res_df


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Zero-dependency markdown table serializer for pandas DataFrames."""
    headers = [str(c) for c in df.columns]
    rows = [[str(val) for val in row] for row in df.itertuples(index=False)]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + row_lines)


def save_benchmark_outputs(summary_df: pd.DataFrame, output_dir: str = "outputs"):
    """Saves benchmark results to CSV, Markdown, and generates comparison plot."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    md_path = os.path.join(output_dir, "benchmark_results.md")
    plot_path = os.path.join(output_dir, "benchmark_comparison.png")

    summary_df.to_csv(csv_path, index=False)

    md_table = dataframe_to_markdown(summary_df)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DataSentry Benchmark Comparison\n\n")
        f.write("Empirical benchmark across 3 public tabular datasets (Adult Census, Sensor Telemetry, Healthcare Vitals) ")
        f.write("evaluating 10 fault types against clean controls.\n\n")
        f.write(md_table)
        f.write("\n")

    # Generate matplotlib comparison plot
    try:
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 6))
        detectors = summary_df["detector"].tolist()
        f1_scores = summary_df["f1_score"].tolist()
        recalls = summary_df["recall"].tolist()
        precisions = summary_df["precision"].tolist()

        x = range(len(detectors))
        width = 0.25

        ax1.bar([i - width for i in x], f1_scores, width=width, label="F1 Score", color="#0f62fe")
        ax1.bar(x, recalls, width=width, label="Recall", color="#198038")
        ax1.bar([i + width for i in x], precisions, width=width, label="Precision", color="#fa4d56")

        ax1.set_xlabel("Detector")
        ax1.set_ylabel("Score")
        ax1.set_title("DataSentry vs Baselines on Injected Fault Benchmark")
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(detectors)
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        ax1.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Warning: Matplotlib plot could not be saved: {e}")
