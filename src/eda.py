"""Reusable dataset profiling and exploratory analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("This starter framework currently expects a .csv file.")
    return pd.read_csv(path)


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Return key structural information about a DataFrame."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_column_names": int(df.columns.duplicated().sum()),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / (1024**2), 3),
    }


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Build one profile row per column."""
    total = len(df)
    rows = []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        rows.append({
            "column": str(col),
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": missing,
            "missing_percent": round((missing / total * 100) if total else 0.0, 2),
            "unique_values": int(series.nunique(dropna=True)),
            "sample_values": ", ".join(map(str, series.dropna().astype(str).head(3).tolist())),
        })
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe().T.reset_index(names="column")


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top category frequencies for non-numeric columns."""
    categorical = df.select_dtypes(exclude="number")
    rows = []
    for col in categorical.columns:
        counts = categorical[col].astype("string").fillna("<MISSING>").value_counts(dropna=False).head(top_n)
        for value, count in counts.items():
            rows.append({
                "column": str(col),
                "value": str(value),
                "count": int(count),
                "percent": round(float(count) / len(df) * 100, 2) if len(df) else 0.0,
            })
    return pd.DataFrame(rows, columns=["column", "value", "count", "percent"])


def iqr_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag numeric values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.

    This function identifies candidate outliers only. It does not remove or change data.
    """
    numeric = df.select_dtypes(include="number")
    rows = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            low, high = q1, q3
        else:
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        for idx, value in df.loc[mask, col].items():
            rows.append({
                "row_index": idx,
                "column": str(col),
                "value": value,
                "lower_bound": low,
                "upper_bound": high,
            })
    return pd.DataFrame(rows, columns=["row_index", "column", "value", "lower_bound", "upper_bound"])


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson correlations among numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return pd.DataFrame()
    return numeric.corr(numeric_only=True)


def build_quality_notes(df: pd.DataFrame, profile: pd.DataFrame, outliers: pd.DataFrame) -> list[str]:
    """Create concise quality observations for the report."""
    notes: list[str] = []
    total_missing = int(df.isna().sum().sum())
    if total_missing:
        notes.append(f"The dataset contains {total_missing:,} missing cell(s). Review the column-level missingness profile before deciding how to handle them.")
    else:
        notes.append("No missing cells were detected by pandas' standard missing-value checks.")
    duplicates = int(df.duplicated().sum())
    if duplicates:
        notes.append(f"{duplicates:,} exact duplicate row(s) were detected. Confirm whether they are accidental before removing them.")
    else:
        notes.append("No exact duplicate rows were detected.")
    if not outliers.empty:
        notes.append(f"The IQR screening rule flagged {len(outliers):,} numeric value(s) as potential outliers. These are flags for investigation, not automatic errors.")
    else:
        notes.append("The IQR screening rule did not flag any numeric values, or the dataset contains no numeric columns.")
    if not profile.empty:
        high_missing = profile.loc[profile["missing_percent"] >= 30, "column"].tolist()
        if high_missing:
            notes.append("Column(s) with at least 30% missing values: " + ", ".join(high_missing) + ". Investigate why these values are missing.")
    return notes
