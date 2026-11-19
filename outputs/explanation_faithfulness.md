# Phase 4: Explanation Layer Faithfulness & Hallucination Results

Evaluation of Grounded LLM Explainer across injected faults on 3 tabular datasets.

| Total Reports Evaluated | Mean Citation Precision | Mean Citation Recall | Mean Column Faithfulness | Mean Numeric Faithfulness | Overall Hallucination Rate | Faithful Explanations Ratio | Average Latency (ms) |
| ----------------------- | ----------------------- | -------------------- | ------------------------ | ------------------------- | -------------------------- | --------------------------- | -------------------- |
| 12                      | 100.00%                 | 100.00%              | 100.00%                  | 100.00%                   | 0.00%                      | 100.00%                     | 0.07                 |

### Detailed Evaluations by Scenario

| dataset           | scenario          | findings_count | cited_count | citation_precision | numeric_faithfulness | hallucination_rate | is_faithful |
| ----------------- | ----------------- | -------------- | ----------- | ------------------ | -------------------- | ------------------ | ----------- |
| adult_census      | CLEAN_CONTROL     | 1              | 1           | 1.0                | 1.0                  | 0.0                | True        |
| adult_census      | MISSINGNESS_SPIKE | 2              | 2           | 1.0                | 1.0                  | 0.0                | True        |
| adult_census      | COVARIATE_SHIFT   | 3              | 3           | 1.0                | 1.0                  | 0.0                | True        |
| adult_census      | DUPLICATE_ROWS    | 2              | 2           | 1.0                | 1.0                  | 0.0                | True        |
| sensor_telemetry  | CLEAN_CONTROL     | 0              | 0           | 1.0                | 1.0                  | 0.0                | True        |
| sensor_telemetry  | MISSINGNESS_SPIKE | 1              | 1           | 1.0                | 1.0                  | 0.0                | True        |
| sensor_telemetry  | COVARIATE_SHIFT   | 1              | 1           | 1.0                | 1.0                  | 0.0                | True        |
| sensor_telemetry  | DUPLICATE_ROWS    | 1              | 1           | 1.0                | 1.0                  | 0.0                | True        |
| healthcare_vitals | CLEAN_CONTROL     | 3              | 3           | 1.0                | 1.0                  | 0.0                | True        |
| healthcare_vitals | MISSINGNESS_SPIKE | 4              | 4           | 1.0                | 1.0                  | 0.0                | True        |
| healthcare_vitals | COVARIATE_SHIFT   | 4              | 4           | 1.0                | 1.0                  | 0.0                | True        |
| healthcare_vitals | DUPLICATE_ROWS    | 4              | 4           | 1.0                | 1.0                  | 0.0                | True        |
