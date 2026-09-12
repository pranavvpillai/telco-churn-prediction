"""
Phase 4: Feature Engineering
churn/scripts/feature_engineering.py

Reads the normalized tables from data/churn.db (built by load_data.py),
joins them back into one analysis-ready frame, and engineers a set of
derived features on top of the raw columns already validated in Phase 3
(statistical_tests.py).

Run:
    python scripts/feature_engineering.py

Output:
    - data/churn.db gets a new table: features
    - outputs/engineered_features.csv (same data, for quick inspection)
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/churn.db")
OUTPUT_CSV = Path("outputs/engineered_features.csv")

# Columns Phase 3 confirmed as real (non-negligible) churn signal.
# Kept here as a single source of truth so downstream modeling scripts
# can import this list instead of re-deriving it.
SIGNIFICANT_MODERATE_OR_LARGER = [
    "tenure",  # large effect (Cohen's d = -0.89)
    "Contract",  # moderate (Cramer's V = 0.41)
    "MonthlyCharges",  # moderate (Cohen's d = 0.47)
    "OnlineSecurity",
    "TechSupport",
    "InternetService",
    "PaymentMethod",
]

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_joined_table(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Join customers + services + billing back into one row-per-customer frame."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run scripts/load_data.py first (Phase 1)."
        )

    con = sqlite3.connect(db_path)
    try:
        query = """
            SELECT c.*, s.*, b.*
            FROM customers c
            JOIN services s ON c.customerID = s.customerID
            JOIN billing b ON c.customerID = b.customerID
        """
        df = pd.read_sql_query(query, con)
    finally:
        con.close()

    # Drop duplicate customerID columns produced by the SELECT * joins,
    # keeping the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket tenure into standard 0-1yr / 1-2yr / 2-4yr / 4yr+ groups."""
    bins = [-1, 12, 24, 48, df["tenure"].max() + 1]
    labels = ["0-1yr", "1-2yr", "2-4yr", "4yr+"]
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=bins, labels=labels)
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count how many of the six add-on services each customer has (value == 'Yes')."""
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
    return df


def add_avg_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly spend per active service (services + base line), avoids div-by-zero."""
    df["avg_charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    return df


def add_charge_to_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges / tenure — flags customers paying a lot relative to how long they've stayed.
    tenure == 0 (brand-new customers) is mapped to NaN rather than divide-by-zero.
    """
    df["charge_to_tenure_ratio"] = df.apply(
        lambda row: row["TotalCharges"] / row["tenure"] if row["tenure"] > 0 else float("nan"),
        axis=1,
    )
    return df


def add_is_month_to_month(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag for the Contract type Phase 3 found to be the strongest categorical driver."""
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_has_internet(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: does the customer have any internet service at all."""
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    return df


def add_senior_and_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Flag senior citizens with no partner and no dependents — a plausible risk segment
    worth checking against the effect-size results, even though SeniorCitizen alone
    only showed a small effect in Phase 3.
    """
    df["senior_and_alone"] = (
        (df["SeniorCitizen"] == 1) & (df["Partner"] == "No") & (df["Dependents"] == "No")
    ).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature engineering pipeline in order."""
    df = add_tenure_bucket(df)
    df = add_service_count(df)
    df = add_avg_charge_per_service(df)
    df = add_charge_to_tenure_ratio(df)
    df = add_is_month_to_month(df)
    df = add_has_internet(df)
    df = add_senior_and_alone(df)
    return df


def save_features(df: pd.DataFrame, db_path: Path = DB_PATH, csv_path: Path = OUTPUT_CSV) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    con = sqlite3.connect(db_path)
    try:
        df.to_sql("features", con, if_exists="replace", index=False)
    finally:
        con.close()


def main() -> None:
    df = load_joined_table()
    print(f"Loaded {len(df)} customers for feature engineering")

    df = engineer_features(df)
    new_cols = [
        "tenure_bucket",
        "num_services",
        "avg_charge_per_service",
        "charge_to_tenure_ratio",
        "is_month_to_month",
        "has_internet",
        "senior_and_alone",
    ]
    print(f"Added {len(new_cols)} engineered features: {', '.join(new_cols)}")

    save_features(df)
    print(f"features table written to {DB_PATH}")
    print(f"CSV copy written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()