# DataSentry — Project Plan & Specification

## 1. Executive Summary & Problem Statement

Machine Learning (ML) systems in production rarely fail because of bad math; they fail because of silent data degradation. Upstream schema changes, broken data pipelines, unexpected missing fields, out-of-bounds sensor readings, and shifting population demographics can corrupt predictions without throwing a single fatal runtime exception.

**DataSentry** is a lightweight, explainable monitoring and diagnostic engine for tabular data. It inspects data across three distinct analytical axes before or during machine learning workflows:
1. **Data-Quality Validation:** Structural integrity, schema conformance, missingness, constraints, and duplicates.
2. **Record-Level Anomaly Detection:** Identification of individual rows that appear statistically unusual or physically implausible relative to reference distributions or explicit heuristic bounds.
3. **Dataset-Level Drift Detection:** Comparison of feature distributions between a trusted baseline ("Reference") and an incoming batch ("Current") to spot distribution shift.

---

## 2. Target Users & User Stories

### Target Personas
* **Data Scientists:** Verifying clean data splits and detecting distribution shift prior to training or fine-tuning models.
* **Machine Learning Engineers (MLEs):** Validating incoming inference batches before feeding features into production models.
* **Data Engineers:** Profiling batches at ingestion boundaries to halt downstream pipelines when catastrophic data corruption occurs.

### User Stories
* **Story 1 (Pre-inference Gate):** *As an ML Engineer*, I want to pass an incoming batch of tabular inference data through an automated check so that malformed records (e.g., negative ages, missing mandatory keys) are isolated and prevented from triggering erroneous model predictions.
* **Story 2 (Distribution Shift Alert):** *As a Data Scientist*, I want to compare this month's customer data against our training reference set using statistical distance metrics, so I can detect whether our feature distributions have shifted significantly and evaluate if retraining is warranted.
* **Story 3 (Explainable Anomaly Inspection):** *As an Auditor/Analyst*, I want to see which specific records in a dataset were flagged as anomalous along with human-readable rationale (which features violated bounds or contributed to the outlier score), rather than an opaque black-box flag.

---

## 3. Minimum Viable Product (MVP) Acceptance Criteria

| ID | Feature Area | Measurable Acceptance Criteria |
|---|---|---|
| **AC-1** | **Data Loading** | Safe ingestion of CSV inputs; non-destructive (never mutates raw source files); reports shape, inferred data types, and file metadata. |
| **AC-2** | **Quality Profiling** | Detects and quantifies: schema discrepancies (missing/unexpected columns), null percentages per column, exact duplicate rows, and value boundary violations. |
| **AC-3** | **Anomaly Detection** | Provides a dual-tier approach: (1) Transparent rule-based bounds (IQR/domain thresholds), and (2) An unsupervised baseline (Isolation Forest). Flags include row indexes and explanation vectors. |
| **AC-4** | **Drift Detection** | Compares Reference vs. Current datasets. Computes two-sample statistical tests / distance metrics for numeric features (e.g., Kolmogorov-Smirnov) and category frequency distance (e.g., Population Stability Index or Chi-Square/Wasserstein) with configurable significance thresholds. |
| **AC-5** | **Structured Reporting** | Consolidates quality, anomaly, and drift metrics into a strict, validated JSON schema (Pydantic models) without non-serializable NumPy/pandas objects. |
| **AC-6** | **Interfaces** | Exposes functionality via a clean FastAPI HTTP API (`/health`, `/profile`, `/analyze`) and an interactive inspection UI (Streamlit or lightweight HTML interface). |
| **AC-7** | **Test Coverage** | Full automated unit and integration test suite with `pytest` covering edge cases: empty data, zero variance, all-null columns, unseen categories, and controlled synthetic drift. |

---

## 4. Known Boundaries & Non-Goals ("What DataSentry is NOT")

1. **Not an automatic data-cleansing magic wand:** DataSentry will **never** silently delete rows, impute arbitrary means, or guess corrections. It is a diagnostic and profiling instrument. Correction policies must be explicit, deliberate, and auditable.
2. **Not proof of model failure:** A statistical drift alert (e.g., $p < 0.05$ on a Kolmogorov-Smirnov test) signifies a change in distribution, but it does **not** prove that model accuracy has degraded. Drift detection is an alert to investigate, not an automated indictment.
3. **Not a distributed big-data engine:** The MVP targets in-memory tabular processing (pandas/NumPy/scikit-learn) suitable for batch payloads (tens to hundreds of thousands of rows). It does not distribute compute across Spark clusters.
4. **An anomaly is not automatically an error:** An anomalous record might be a high-value customer, a rare fraud event, or a valid edge case. Anomaly flags must be reviewed in context, not discarded as corrupt data.
