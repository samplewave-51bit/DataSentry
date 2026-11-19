"""
Phase 1: Run Fault-Injection Benchmark

Executes DataSentry evaluation harness against baselines:
- Naive KS-test
- Great Expectations
- Evidently

Saves outputs to outputs/benchmark_results.csv, outputs/benchmark_results.md,
and outputs/benchmark_comparison.png.
"""

from datasentry.evaluation.benchmark import BenchmarkRunner, save_benchmark_outputs


def main():
    print("=" * 60)
    print("Running DataSentry Phase 1 Benchmark Harness")
    print("Evaluating: Adult Census, Sensor Telemetry, Healthcare Vitals")
    print("Baselines: DataSentry, Evidently, Great Expectations, Naive KS")
    print("=" * 60)

    runner = BenchmarkRunner(random_state=42)
    summary_df = runner.run_benchmark(n_runs_per_fault=5)

    print("\nBenchmark Results Summary:")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    print("-" * 60)

    save_benchmark_outputs(summary_df, output_dir="outputs")
    print("\nOutputs saved:")
    print("- outputs/benchmark_results.csv")
    print("- outputs/benchmark_results.md")
    print("- outputs/benchmark_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
