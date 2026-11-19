# DataSentry Benchmark Comparison

Empirical benchmark across 3 public tabular datasets (Adult Census, Sensor Telemetry, Healthcare Vitals) evaluating 10 fault types against clean controls.

| detector           | trials | true_positives | false_positives | true_negatives | false_negatives | precision | recall | f1_score | f1_std | precision_std | recall_std | false_alarm_rate | avg_latency_ms |
| ------------------ | ------ | -------------- | --------------- | -------------- | --------------- | --------- | ------ | -------- | ------ | ------------- | ---------- | ---------------- | -------------- |
| datasentry         | 165    | 120            | 3               | 12             | 30              | 0.9756    | 0.8    | 0.8791   | 0.0142 | 0.0346        | 0.0        | 0.2              | 302.31         |
| great_expectations | 165    | 70             | 0               | 15             | 80              | 1.0       | 0.4667 | 0.6364   | 0.0    | 0.0           | 0.0        | 0.0              | 4.15           |
| evidently          | 165    | 60             | 0               | 15             | 90              | 1.0       | 0.4    | 0.5714   | 0.0    | 0.0           | 0.0        | 0.0              | 65.49          |
| naive_ks           | 165    | 30             | 0               | 15             | 120             | 1.0       | 0.2    | 0.3333   | 0.0    | 0.0           | 0.0        | 0.0              | 6.97           |
