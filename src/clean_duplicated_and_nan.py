"""
Clean duplicated rows and inspect missing values in the project risk dataset.

- Loads the raw CSV from data/raw/project_risk_raw_dataset.csv
- Drops duplicated rows (keeps the first occurrence)
- Detects missing values (NaN/None/empty/N/A-like tokens)
- Saves the cleaned CSV to data/processed/project_risk_clean.csv
- Writes a text report to info/clean_report.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


# --- Repository-relative paths ---
RAW_CSV_PATH = Path("data/raw/project_risk_raw_dataset.csv")
PROCESSED_DIR = Path("data/processed")
CLEAN_CSV_PATH = PROCESSED_DIR / "project_risk_clean.csv"
INFO_DIR = Path("info")
REPORT_PATH = INFO_DIR / "clean_report.txt"

# Tokens to treat as missing in addition to pandas defaults
EXTRA_NA_VALUES: Iterable[str] = ("", " ", "NA", "N/A", "NULL", "Null", "null")


@dataclass(frozen=True)
class CleanSummary:
    """Summary of the cleaning process."""
    raw_rows: int
    raw_cols: int
    duplicate_rows_removed: int
    cleaned_rows: int
    cleaned_cols: int
    total_missing_values: int
    missing_by_column: Dict[str, int]


def ensure_directories(*paths: Path) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path, extra_na: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame, interpreting common 'NA-like' tokens as missing.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    extra_na : Optional[Iterable[str]]
        Additional strings to interpret as NA.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    na_values = set(extra_na or [])
    df = pd.read_csv(
        path,
        na_values=na_values,
        keep_default_na=True,
        # If you know there is no index column in the file:
        # index_col=False
    )
    return df


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop duplicated rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (cleaned_df, num_duplicates_removed)
    """
    before = len(df)
    cleaned = df.drop_duplicates(keep="first", ignore_index=True)
    removed = before - len(cleaned)
    return cleaned, removed


def compute_missing(df: pd.DataFrame) -> tuple[int, Dict[str, int]]:
    """
    Compute total and per-column missing (NaN) values.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[int, Dict[str, int]]
        (total_missing, missing_by_column)
    """
    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(sum(missing_by_col.values()))
    return total_missing, missing_by_col


def write_report(path: Path, summary: CleanSummary) -> None:
    """
    Write a plain-text report with cleaning details.

    Parameters
    ----------
    path : Path
        Destination path for the report.
    summary : CleanSummary
        Summary data to write.
    """
    lines = []
    lines.append("=== Data Cleaning Report ===")
    lines.append("")
    lines.append(f"Raw shape                : {summary.raw_rows} rows x {summary.raw_cols} cols")
    lines.append(f"Duplicates removed       : {summary.duplicate_rows_removed}")
    lines.append(f"Cleaned shape            : {summary.cleaned_rows} rows x {summary.cleaned_cols} cols")
    lines.append("")
    lines.append(f"Total missing values     : {summary.total_missing_values}")
    lines.append("Missing values by column :")
    # Align column counts nicely
    max_name_len = max((len(c) for c in summary.missing_by_column.keys()), default=0)
    for col, cnt in sorted(summary.missing_by_column.items()):
        lines.append(f"  - {col:<{max_name_len}} : {cnt}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def save_clean_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save the cleaned DataFrame as CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : Path
        Output CSV path.
    """
    df.to_csv(path, index=False)


def main() -> None:
    """Run the cleaning pipeline end-to-end."""
    # Ensure output directories exist
    ensure_directories(PROCESSED_DIR, INFO_DIR)

    # Load raw dataset
    df_raw = load_dataset(RAW_CSV_PATH, extra_na=EXTRA_NA_VALUES)
    raw_rows, raw_cols = df_raw.shape

    # Drop duplicates
    df_clean, dup_removed = drop_duplicates(df_raw)

    # Compute missing values on the cleaned data
    total_missing, missing_by_col = compute_missing(df_clean)

    # Persist outputs
    save_clean_csv(df_clean, CLEAN_CSV_PATH)

    summary = CleanSummary(
        raw_rows=raw_rows,
        raw_cols=raw_cols,
        duplicate_rows_removed=dup_removed,
        cleaned_rows=df_clean.shape[0],
        cleaned_cols=df_clean.shape[1],
        total_missing_values=total_missing,
        missing_by_column=missing_by_col,
    )
    write_report(REPORT_PATH, summary)


if __name__ == "__main__":
    main()
