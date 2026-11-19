# DataSentry — Architectural Design Decisions

This document records the foundational architectural decisions made for DataSentry, including their technical context, alternatives considered, and trade-offs.

---

## Decision 1: Strict Separation of the Analytical Triad
* **Context:** Many data platforms roll missing values, outlier scores, and drift into a single composite "Data Health Score" (e.g., "Dataset Health: 82%").
* **Decision:** Keep Data Quality Validation, Record-Level Anomaly Detection, and Dataset Drift Detection strictly separated in code, schema, and reporting.
* **Rationale:** 
  - *Data Quality* evaluates absolute deterministic constraints (e.g., "Age cannot be negative").
  - *Anomaly Detection* evaluates row-level relative likelihood under an assumed distribution (e.g., "This combination of salary and age is rare").
  - *Drift Detection* evaluates population-level statistical divergence across time/batches (e.g., "The average age shifted from 32 to 45").
  - Merging these into a single metric obscures root causes. A dataset could have 100% valid quality yet have severe drift, or have zero drift while harboring dangerous isolated anomalies.
* **Trade-off:** Slightly more verbose API reports, but eliminates ambiguity.

---

## Decision 2: Passive Diagnostic Profiling vs. Active Auto-Mutation
* **Context:** It is tempting to write "cleansing" scripts that automatically fill missing values with column medians or drop rows beyond 3 standard deviations.
* **Decision:** DataSentry operates primarily as a non-destructive diagnostic sensor. It flags, measures, and reports, but does not mutate source data unless explicitly instructed by a documented, task-specific pipeline policy.
* **Rationale:** Silent imputation or row deletion causes data leakage, distorts statistical variance, and can destroy genuine edge cases (such as actual fraud signals).
* **Trade-off:** Downstream applications must consume the diagnostic report and apply domain-appropriate handling rather than relying on DataSentry to "fix" everything automatically.

---

## Decision 3: Separation of Core Logic from API and Presentation Layers
* **Context:** Analytics tools often bind data checks tightly to web framework callbacks (e.g., inside FastAPI route handlers or Streamlit script loops).
* **Decision:** Implement all data loading, validation, profiling, and modeling as pure, modular Python functions/classes with zero web dependencies. The API (`FastAPI`) and UI (`Streamlit` / HTML) will act strictly as thin adapters over the core package.
* **Rationale:** Pure Python modules are straightforward to unit-test with `pytest`, run in CI/CD pipelines, and embed into batch scripts without spinning up web servers.
* **Trade-off:** Requires defining explicit data transfer objects (DTOs / Pydantic schemas) to pass state cleanly between the engine and the presentation layer.

---

## Decision 4: Synthetic Fault Injection & Controlled Drift for Verification
* **Context:** Real-world datasets often lack verified ground-truth labels for anomalies or subtle drift, making objective evaluation of detection systems difficult.
* **Decision:** We evaluate our anomaly and drift algorithms using deliberate, reproducible synthetic fault injection (controlled corruption and shifts) on top of clean baselines.
* **Rationale:** With synthetic injection, ground truth is known with 100% certainty (we know precisely which rows were corrupted and which features were shifted). This allows empirical measurement of True Positives, False Positives, and detection sensitivity.
* **Trade-off:** Synthetic anomalies are sometimes more distinct than messy real-world noise, so performance on synthetic benchmarks must be explicitly qualified as a baseline capability test, not proof of real-world perfection.
