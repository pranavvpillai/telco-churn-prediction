"""Load the Telco Customer Churn CSV into a normalized SQLite database.

Splits the single flat CSV into three related tables -- customers,
services, billing -- so the project involves real joins rather than
querying one flat table dressed up as SQL.

Usage:
    python scripts/load_data.py [path_to_csv]

Defaults to data/Telco-Customer-Churn.csv if no path is given; falls back
to the synthetic sample if the real file isn't present yet.
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "churn.db"

REAL_CSV = DATA_DIR / "Telco-Customer-Churn.csv"
SAMPLE_CSV = DATA_DIR / "Telco-Customer-Churn-SAMPLE.csv"


def resolve_csv_path(cli_arg: str | None) -> Path:
    if cli_arg:
        path = Path(cli_arg)
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}")
        return path
    if REAL_CSV.exists():
        return REAL_CSV
    if SAMPLE_CSV.exists():
        print(f"NOTE: real dataset not found at {REAL_CSV.name} -- using synthetic "
              f"sample ({SAMPLE_CSV.name}) instead. Drop in the real CSV and rerun "
              f"for actual results.")
        return SAMPLE_CSV
    raise FileNotFoundError(
        f"No dataset found. Place the real CSV at {REAL_CSV} "
        f"(download from Kaggle: blastchar/telco-customer-churn)."
    )


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues in the raw Telco dataset."""
    df = df.copy()
    # TotalCharges is stored as text and has blank strings for tenure=0
    # customers (brand new, no charges yet) -- coerce to numeric, NaN for blanks.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def build_tables(df: pd.DataFrame):
    """Split the flat dataframe into three normalized tables."""
    customers = df[[
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "Contract", "PaperlessBilling", "PaymentMethod", "Churn",
    ]].copy()

    services = df[[
        "customerID", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]].copy()

    billing = df[[
        "customerID", "MonthlyCharges", "TotalCharges",
    ]].copy()

    return customers, services, billing


def load_to_sqlite(customers, services, billing, db_path: Path):
    if db_path.exists():
        db_path.unlink()  # fresh load each run
    conn = sqlite3.connect(db_path)
    try:
        customers.to_sql("customers", conn, index=False)
        services.to_sql("services", conn, index=False)
        billing.to_sql("billing", conn, index=False)
        conn.execute("CREATE INDEX idx_customers_id ON customers(customerID)")
        conn.execute("CREATE INDEX idx_services_id ON services(customerID)")
        conn.execute("CREATE INDEX idx_billing_id ON billing(customerID)")
        conn.commit()
    finally:
        conn.close()


def main():
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = resolve_csv_path(cli_arg)

    df = pd.read_csv(csv_path)
    df = clean_raw(df)
    customers, services, billing = build_tables(df)
    load_to_sqlite(customers, services, billing, DB_PATH)

    print(f"Loaded {len(df)} customers from {csv_path.name}")
    print(f"  customers table: {customers.shape}")
    print(f"  services table:  {services.shape}")
    print(f"  billing table:   {billing.shape}")
    print(f"Database written to {DB_PATH}")


if __name__ == "__main__":
    main()
