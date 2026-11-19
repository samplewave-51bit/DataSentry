"""
DataSentry Evaluation - Fault Injection Module

Corrupts clean tabular datasets with labeled, reproducible faults:
1. Schema Change (dropped, renamed, retyped)
2. Missingness Spike
3. Unit Error (e.g. Celsius to Fahrenheit)
4. Out-of-Range Values
5. Duplicate Rows
6. Covariate Shift
7. Label Shift
8. Correlated-Feature Breakage
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from datasentry.schemas import Severity


class FaultType(str, Enum):
    SCHEMA_DROP = "schema_drop"
    SCHEMA_RENAME = "schema_rename"
    SCHEMA_RETYPE = "schema_retype"
    MISSINGNESS_SPIKE = "missingness_spike"
    UNIT_ERROR = "unit_error"
    OUT_OF_RANGE = "out_of_range"
    DUPLICATE_ROWS = "duplicate_rows"
    COVARIATE_SHIFT = "covariate_shift"
    LABEL_SHIFT = "label_shift"
    CORRELATED_FEATURE_BREAKAGE = "correlated_feature_breakage"


@dataclass
class FaultInjectionResult:
    fault_type: FaultType
    severity: Severity
    corrupted_df: pd.DataFrame
    target_column: Optional[str] = None
    target_rows: List[int] = field(default_factory=list)
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)


class FaultInjector:
    """Injects reproducible, labeled faults into clean DataFrames."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def inject_schema_drop(
        self, df: pd.DataFrame, column: str, severity: Severity = Severity.CRITICAL
    ) -> FaultInjectionResult:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        corrupted = df.drop(columns=[column]).copy()
        return FaultInjectionResult(
            fault_type=FaultType.SCHEMA_DROP,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            description=f"Dropped mandatory column '{column}'.",
        )

    def inject_schema_rename(
        self,
        df: pd.DataFrame,
        old_column: str,
        new_column: Optional[str] = None,
        severity: Severity = Severity.CRITICAL,
    ) -> FaultInjectionResult:
        if old_column not in df.columns:
            raise ValueError(f"Column '{old_column}' not found in DataFrame.")
        new_col_name = new_column or f"{old_column}_renamed"
        corrupted = df.rename(columns={old_column: new_col_name}).copy()
        return FaultInjectionResult(
            fault_type=FaultType.SCHEMA_RENAME,
            severity=severity,
            corrupted_df=corrupted,
            target_column=old_column,
            description=f"Renamed column '{old_column}' to '{new_col_name}'.",
            parameters={"new_column": new_col_name},
        )

    def inject_schema_retype(
        self, df: pd.DataFrame, column: str, severity: Severity = Severity.HIGH
    ) -> FaultInjectionResult:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        corrupted = df.copy()
        # Convert numeric to string or vice versa
        if pd.api.types.is_numeric_dtype(corrupted[column]):
            corrupted[column] = corrupted[column].astype(str) + "_str"
        else:
            corrupted[column] = np.arange(len(corrupted), dtype=float)
        return FaultInjectionResult(
            fault_type=FaultType.SCHEMA_RETYPE,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            description=f"Altered data type of column '{column}'.",
        )

    def inject_missingness_spike(
        self,
        df: pd.DataFrame,
        column: str,
        missing_rate: float = 0.40,
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        corrupted = df.copy()
        n_rows = len(df)
        n_missing = int(n_rows * missing_rate)
        target_labels = self.rng.choice(corrupted.index, size=n_missing, replace=False).tolist()
        corrupted.loc[target_labels, column] = np.nan
        return FaultInjectionResult(
            fault_type=FaultType.MISSINGNESS_SPIKE,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            target_rows=[int(idx) if isinstance(idx, (int, np.integer)) else i for i, idx in enumerate(target_labels)],
            description=f"Spiked missingness in '{column}' by {missing_rate:.1%}.",
            parameters={"missing_rate": missing_rate},
        )

    def inject_unit_error(
        self,
        df: pd.DataFrame,
        column: str,
        conversion: str = "celsius_to_fahrenheit",
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be a numeric column.")
        corrupted = df.copy()
        if conversion == "celsius_to_fahrenheit":
            corrupted[column] = corrupted[column] * 1.8 + 32.0
        elif conversion == "kg_to_lbs":
            corrupted[column] = corrupted[column] * 2.20462
        else:
            corrupted[column] = corrupted[column] * 100.0

        return FaultInjectionResult(
            fault_type=FaultType.UNIT_ERROR,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            target_rows=list(range(len(df))),
            description=f"Injected unit error '{conversion}' on column '{column}'.",
            parameters={"conversion": conversion},
        )

    def inject_out_of_range(
        self,
        df: pd.DataFrame,
        column: str,
        fraction: float = 0.05,
        scale_factor: float = 10.0,
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be a numeric column.")
        corrupted = df.copy()
        n_rows = len(df)
        n_outliers = max(1, int(n_rows * fraction))
        target_labels = self.rng.choice(corrupted.index, size=n_outliers, replace=False).tolist()

        col_std = corrupted[column].std()
        col_mean = corrupted[column].mean()
        # Inject extreme values in both directions
        for i, idx in enumerate(target_labels):
            sign = 1 if i % 2 == 0 else -1
            corrupted.loc[idx, column] = col_mean + sign * scale_factor * (col_std + 1.0)

        return FaultInjectionResult(
            fault_type=FaultType.OUT_OF_RANGE,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            target_rows=[int(idx) if isinstance(idx, (int, np.integer)) else i for i, idx in enumerate(target_labels)],
            description=f"Injected {n_outliers} out-of-range records in '{column}'.",
            parameters={"fraction": fraction, "scale_factor": scale_factor},
        )

    def inject_duplicate_rows(
        self,
        df: pd.DataFrame,
        fraction: float = 0.10,
        severity: Severity = Severity.MEDIUM,
    ) -> FaultInjectionResult:
        n_rows = len(df)
        n_dups = max(1, int(n_rows * fraction))
        dup_indices = self.rng.choice(df.index, size=n_dups, replace=False).tolist()
        duplicates = df.loc[dup_indices].copy()
        corrupted = pd.concat([df, duplicates], ignore_index=True)
        return FaultInjectionResult(
            fault_type=FaultType.DUPLICATE_ROWS,
            severity=severity,
            corrupted_df=corrupted,
            target_rows=list(range(n_rows, n_rows + n_dups)),
            description=f"Injected {n_dups} duplicate rows ({fraction:.1%}).",
            parameters={"fraction": fraction},
        )

    def inject_covariate_shift(
        self,
        df: pd.DataFrame,
        column: str,
        shift_std: float = 3.0,
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' must be a numeric column.")
        corrupted = df.copy()
        col_std = corrupted[column].std()
        corrupted[column] = corrupted[column] + shift_std * col_std
        return FaultInjectionResult(
            fault_type=FaultType.COVARIATE_SHIFT,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            target_rows=list(range(len(df))),
            description=f"Shifted distribution of '{column}' by +{shift_std} sigma.",
            parameters={"shift_std": shift_std},
        )

    def inject_label_shift(
        self,
        df: pd.DataFrame,
        column: str,
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")
        corrupted = df.copy()
        # Invert category frequencies or flip binary labels
        unique_vals = list(corrupted[column].dropna().unique())
        if len(unique_vals) >= 2:
            mapping = {unique_vals[i]: unique_vals[(i + 1) % len(unique_vals)] for i in range(len(unique_vals))}
            corrupted[column] = corrupted[column].map(mapping).fillna(corrupted[column])

        return FaultInjectionResult(
            fault_type=FaultType.LABEL_SHIFT,
            severity=severity,
            corrupted_df=corrupted,
            target_column=column,
            target_rows=list(range(len(df))),
            description=f"Permuted category/label frequencies in '{column}'.",
        )

    def inject_correlated_feature_breakage(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
        severity: Severity = Severity.HIGH,
    ) -> FaultInjectionResult:
        if col_a not in df.columns or col_b not in df.columns:
            raise ValueError(f"Columns '{col_a}' and '{col_b}' must be in DataFrame.")
        corrupted = df.copy()
        # Break correlation by shuffling col_b independently
        shuffled_vals = self.rng.permutation(corrupted[col_b].values)
        corrupted[col_b] = shuffled_vals

        return FaultInjectionResult(
            fault_type=FaultType.CORRELATED_FEATURE_BREAKAGE,
            severity=severity,
            corrupted_df=corrupted,
            target_column=col_b,
            target_rows=list(range(len(df))),
            description=f"Broke correlation between '{col_a}' and '{col_b}' via random permutation.",
            parameters={"col_a": col_a, "col_b": col_b},
        )
