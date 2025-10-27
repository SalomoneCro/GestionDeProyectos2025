"""
TG2-9 | Export dataset to Parquet and generate checksum.

- Loads data/processed/project_risk_clean.csv
- Saves as data/processed/train_ready.parquet
- Computes SHA-256 checksum
- Writes summary report to info/dataset_checksum.txt
"""

import pandas as pd
import hashlib
from pathlib import Path

CSV_PATH = Path("data/processed/project_risk_clean.csv")
PARQUET_PATH = Path("data/processed/train_ready.parquet")
CHECKSUM_PATH = Path("info/dataset_checksum.txt")

def export_and_checksum():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"File not found: {CSV_PATH} / please run the cleaning pipeline first.")

    # Load the cleaned CSV
    df = pd.read_csv(CSV_PATH)

    # Save to Parquet format
    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)

    # Compute checksum
    checksum = hashlib.sha256(PARQUET_PATH.read_bytes()).hexdigest()

    # Write summary report
    CHECKSUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUM_PATH.write_text(
        (
            "=== Dataset Parquet Export ===\n\n"
            f"Source CSV : {CSV_PATH.name}\n"
            f"Output file: {PARQUET_PATH.name}\n"
            f"Rows       : {len(df)}\n"
            f"Columns    : {len(df.columns)}\n"
            f"Checksum (SHA-256): {checksum}\n"
        ),
        encoding="utf-8",
    )

    print("Parquet exported & checksum generated successfully.")

if __name__ == "__main__":
    export_and_checksum()
