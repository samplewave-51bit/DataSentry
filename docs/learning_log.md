# DataSentry — Engineering & Learning Log

This log documents daily progress, architecture decisions, experimental findings, edge cases, and personal technical reflections throughout the 20-day engineering plan.

---

## Day 1 — Problem Definition & Measurable Success
Date: 2026-10-18
Planned effort: 60 min
Actual effort: [Fill in: e.g., 60 min]

### Goal
Define the scope, user personas, MVP acceptance criteria, analytical boundaries, and non-goals for DataSentry before writing code.

### What I implemented
Files created or changed:
- `docs/project_plan.md`: Problem statement, target personas, user stories, MVP acceptance criteria, and non-goals.
- `docs/design_decisions.md`: Foundational decisions on analytical separation, non-destructive profiling, modular architecture, and synthetic evaluation.
- `docs/learning_log.md`: Learning log tracking framework initialized.

Important concepts codified:
- The distinction between Data Quality, Record Anomalies, and Dataset Drift.
- Passive diagnostic profiling vs. mutative auto-cleaning.

Commands run:
- Verified repository status and file creation.

### What I learned
Concepts:
- [Fill in your reflection: e.g., Why data quality, anomaly detection, and drift detection must be evaluated separately.]

Why the chosen approach makes sense:
- [Fill in your thoughts on why building a diagnostic tool without silent data mutation prevents catastrophic data leakage in production.]

Alternatives considered:
- [Fill in: e.g., Single composite health score vs. three distinct diagnostic axes.]

### Evidence
Tests and results:
- Verified that `docs/project_plan.md` and `docs/design_decisions.md` exist and contain testable criteria.

### Edge cases
| Case | Risk | Detection | Handling | Remaining Limitation |
|---|---|---|---|---|
| Ambiguous requirements | Building features that solve the wrong problem | Comparing implementation against written Acceptance Criteria | Explicitly documenting non-goals | Real-world edge cases may require evolving specs |
| Conflating drift with errors | Discarding valid new trends as bad data | Separating statistical shift from schema/value violations | Alert as drift without deleting rows | Cannot determine business impact without domain labels |

### Decisions
Decision:
- Keep the Triad (Quality, Anomaly, Drift) strictly independent.
Reason:
- They operate at different granularities (cell/column constraint vs. row multivariate outlier vs. dataset population shift).
Trade-off:
- Slightly more complex reporting schema, but clear root-cause attribution.

### Questions to revisit
- Which public tabular dataset will best showcase mixed numeric/categorical features and realistic missingness on Day 3?
- What distance metrics for drift (e.g., KS test vs. Wasserstein vs. PSI) balance interpretability and sample size sensitivity?

### My explanation in my own words
[Write 3–5 sentences explaining today's work and core principles without copying text directly.]
