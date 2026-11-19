"""
DataSentry Core - Axis 1: Data-Quality Validation

Evaluates structural integrity, schema conformance, missingness,
boundary constraints, and duplicate rows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from datasentry.schemas import (
    AxisType,
    ColumnQualityMetric,
    Finding,
    QualityReport,
    Severity,
)


class QualityValidator:
    """Non-destructive data-quality validator for tabular data."""

    def __init__(
        self,
        schema: Optional[Dict[str, str]] = None,
        constraints: Optional[Dict[str, Dict[str, Any]]] = None,
        null_threshold: float = 0.10,
        duplicate_threshold: float = 0.01,
    ):
        """
        Args:
            schema: Mapping of column name to expected broad type ("numeric", "string", "datetime", "boolean").
            constraints: Mapping of column name to constraints (e.g. {"min": 0, "max": 120, "allowed_values": [...]}).
            null_threshold: Warning threshold fraction for missing values (default: 0.10).
            duplicate_threshold: Warning threshold fraction for duplicate rows (default: 0.01).
        """
        self.schema = schema or {}
        self.constraints = constraints or {}
        self.null_threshold = null_threshold
        self.duplicate_threshold = duplicate_threshold

    def validate(self, df: pd.DataFrame) -> QualityReport:
        """Execute quality validation suite on df."""
        if df.empty:
            finding = Finding(
                finding_id="FND-QUAL-001",
                axis=AxisType.QUALITY,
                severity=Severity.CRITICAL,
                metric_name="row_count",
                metric_value=0.0,
                threshold=1.0,
                description="Dataset is completely empty (0 rows).",
            )
            return QualityReport(
                passed=False,
                total_rows=0,
                total_columns=len(df.columns),
                duplicate_rows=0,
                duplicate_percentage=0.0,
                schema_matched=False,
                schema_violations=["Empty dataset."],
                column_metrics={},
                findings=[finding],
            )

        total_rows = len(df)
        total_columns = len(df.columns)
        findings: List[Finding] = []
        schema_violations: List[str] = []

        # 1. Duplicate detection
        duplicate_count = int(df.duplicated().sum())
        duplicate_pct = float(duplicate_count / total_rows)
        if duplicate_pct > self.duplicate_threshold:
            severity = Severity.HIGH if duplicate_pct > 0.05 else Severity.MEDIUM
            findings.append(
                Finding(
                    finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                    axis=AxisType.QUALITY,
                    severity=severity,
                    metric_name="duplicate_ratio",
                    metric_value=duplicate_pct,
                    threshold=self.duplicate_threshold,
                    description=f"Duplicate rows detected: {duplicate_count} rows ({duplicate_pct:.2%}).",
                )
            )

        # 2. Schema validation
        if self.schema:
            expected_cols = set(self.schema.keys())
            found_cols = set(df.columns)

            missing_cols = expected_cols - found_cols
            for col in missing_cols:
                msg = f"Missing mandatory schema column: '{col}'"
                schema_violations.append(msg)
                findings.append(
                    Finding(
                        finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                        axis=AxisType.QUALITY,
                        severity=Severity.CRITICAL,
                        column=col,
                        metric_name="schema_presence",
                        metric_value=0.0,
                        threshold=1.0,
                        description=msg,
                    )
                )

            extra_cols = found_cols - expected_cols
            for col in extra_cols:
                msg = f"Unexpected extra column found in batch: '{col}'"
                schema_violations.append(msg)
                findings.append(
                    Finding(
                        finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                        axis=AxisType.QUALITY,
                        severity=Severity.LOW,
                        column=col,
                        metric_name="schema_unexpected",
                        metric_value=1.0,
                        threshold=0.0,
                        description=msg,
                    )
                )

            # Check expected broad types
            for col, expected_type in self.schema.items():
                if col in df.columns:
                    actual_dtype = df[col].dtype
                    type_ok = self._check_dtype(df[col], expected_type)
                    if not type_ok:
                        msg = f"Type mismatch for column '{col}': expected '{expected_type}', found '{actual_dtype}'."
                        schema_violations.append(msg)
                        findings.append(
                            Finding(
                                finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                                axis=AxisType.QUALITY,
                                severity=Severity.HIGH,
                                column=col,
                                metric_name="schema_type",
                                metric_value=0.0,
                                threshold=1.0,
                                description=msg,
                            )
                        )

        # 3. Column-level metrics & constraints
        col_metrics: Dict[str, ColumnQualityMetric] = {}
        for col in df.columns:
            s = df[col]
            null_count = int(s.isna().sum())
            null_pct = float(null_count / total_rows)
            unique_count = int(s.nunique(dropna=True))
            is_constant = bool(unique_count <= 1 and null_count < total_rows)
            violations: List[str] = []
            out_of_bounds = 0

            # Missingness check
            if null_pct > self.null_threshold:
                sev = Severity.HIGH if null_pct > 0.30 else Severity.MEDIUM
                findings.append(
                    Finding(
                        finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                        axis=AxisType.QUALITY,
                        severity=sev,
                        column=col,
                        metric_name="null_ratio",
                        metric_value=null_pct,
                        threshold=self.null_threshold,
                        description=f"High missingness in '{col}': {null_count} nulls ({null_pct:.2%}).",
                    )
                )

            # Constant feature check
            if is_constant and total_rows > 1:
                findings.append(
                    Finding(
                        finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                        axis=AxisType.QUALITY,
                        severity=Severity.LOW,
                        column=col,
                        metric_name="variance",
                        metric_value=0.0,
                        threshold=1.0,
                        description=f"Column '{col}' is constant (zero variance across {total_rows} rows).",
                    )
                )

            # Constraints check
            if col in self.constraints:
                c_spec = self.constraints[col]
                if pd.api.types.is_numeric_dtype(s):
                    clean_s = s.dropna()
                    if "min" in c_spec:
                        min_viol = clean_s < c_spec["min"]
                        if min_viol.any():
                            cnt = int(min_viol.sum())
                            out_of_bounds += cnt
                            violations.append(f"Values below min ({c_spec['min']}): {cnt} rows")
                    if "max" in c_spec:
                        max_viol = clean_s > c_spec["max"]
                        if max_viol.any():
                            cnt = int(max_viol.sum())
                            out_of_bounds += cnt
                            violations.append(f"Values above max ({c_spec['max']}): {cnt} rows")

                if "allowed_values" in c_spec:
                    allowed = set(c_spec["allowed_values"])
                    clean_s = s.dropna()
                    disallowed = ~clean_s.isin(allowed)
                    if disallowed.any():
                        cnt = int(disallowed.sum())
                        out_of_bounds += cnt
                        violations.append(f"Values outside allowed categories: {cnt} rows")

                if out_of_bounds > 0:
                    sev = Severity.HIGH if (out_of_bounds / total_rows) > 0.05 else Severity.MEDIUM
                    findings.append(
                        Finding(
                            finding_id=f"FND-QUAL-{len(findings) + 1:03d}",
                            axis=AxisType.QUALITY,
                            severity=sev,
                            column=col,
                            metric_name="constraint_violations",
                            metric_value=float(out_of_bounds),
                            threshold=0.0,
                            description=f"Constraint violations on '{col}': {'; '.join(violations)}.",
                        )
                    )

            col_metrics[col] = ColumnQualityMetric(
                column=col,
                dtype=str(s.dtype),
                null_count=null_count,
                null_percentage=null_pct,
                unique_count=unique_count,
                is_constant=is_constant,
                out_of_bounds_count=out_of_bounds,
                violations=violations,
            )

        passed = len(schema_violations) == 0 and not any(f.severity == Severity.CRITICAL for f in findings)

        return QualityReport(
            passed=passed,
            total_rows=total_rows,
            total_columns=total_columns,
            duplicate_rows=duplicate_count,
            duplicate_percentage=duplicate_pct,
            schema_matched=len(schema_violations) == 0,
            schema_violations=schema_violations,
            column_metrics=col_metrics,
            findings=findings,
        )

    def _check_dtype(self, series: pd.Series, expected: str) -> bool:
        expected = expected.lower()
        if expected in ("numeric", "number", "float", "int"):
            return bool(pd.api.types.is_numeric_dtype(series))
        if expected in ("string", "str", "object", "category"):
            return bool(pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series))
        if expected in ("datetime", "timestamp"):
            return bool(pd.api.types.is_datetime64_any_dtype(series))
        if expected in ("bool", "boolean"):
            return bool(pd.api.types.is_bool_dtype(series))
        return True
