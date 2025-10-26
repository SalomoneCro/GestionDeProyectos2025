"""
TG2-10 | Validate dataset schema and data types.

- Loads data/processed/train_ready.parquet
- Compares columns with the expected schema
- Reports missing/unexpected columns and dtypes
- Writes a markdown report to reports/schema_validation.md
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/train_ready.parquet")
REPORT_PATH = Path("reports/schema_validation.md")

# Expected columns (aligned with EDA and data dictionary)
EXPECTED_COLUMNS = [
    "Project_ID", "Project_Type", "Team_Size", "Project_Budget_USD",
    "Estimated_Timeline_Months", "Complexity_Score", "Stakeholder_Count",
    "Methodology_Used", "Team_Experience_Level", "Past_Similar_Projects",
    "External_Dependencies_Count", "Change_Request_Frequency",
    "Project_Phase", "Requirement_Stability", "Team_Turnover_Rate",
    "Vendor_Reliability_Score", "Historical_Risk_Incidents",
    "Communication_Frequency", "Regulatory_Compliance_Level",
    "Technology_Familiarity", "Geographical_Distribution",
    "Stakeholder_Engagement_Level", "Schedule_Pressure",
    "Budget_Utilization_Rate", "Executive_Sponsorship", "Funding_Source",
    "Market_Volatility", "Integration_Complexity", "Resource_Availability",
    "Priority_Level", "Organizational_Change_Frequency",
    "Cross_Functional_Dependencies", "Previous_Delivery_Success_Rate",
    "Technical_Debt_Level", "Project_Manager_Experience",
    "Org_Process_Maturity", "Data_Security_Requirements",
    "Key_Stakeholder_Availability", "Tech_Environment_Stability",
    "Contract_Type", "Resource_Contention_Level", "Industry_Volatility",
    "Client_Experience_Level", "Change_Control_Maturity",
    "Risk_Management_Maturity", "Team_Colocation", "Documentation_Quality",
    "Project_Start_Month", "Current_Phase_Duration_Months",
    "Seasonal_Risk_Factor", "Risk_Level"
]

def validate_schema():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run TG2-9 first.")

    df = pd.read_parquet(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra   = [c for c in df.columns if c not in EXPECTED_COLUMNS]

    dtypes_str = df.dtypes.to_string()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join([
            "# Schema Validation Report",
            "",
            f"**File:** {DATA_PATH.name}",
            f"**Rows:** {len(df)}",
            f"**Columns:** {len(df.columns)}",
            "",
            "## Missing Columns",
            ("- " + "\n- ".join(missing)) if missing else "None",
            "",
            "## Unexpected Columns",
            ("- " + "\n- ".join(extra)) if extra else "None",
            "",
            "## pandas dtypes",
            "```",
            dtypes_str,
            "```",
        ]),
        encoding="utf-8",
    )
    print(f"Schema report written to {REPORT_PATH}")

if __name__ == "__main__":
    validate_schema()
