"""
Phase 7: Dashboard data export (for Power BI)
churn/scripts/generate_dashboard_data.py

Power BI has no native SQLite connector, so this script reads the
`features` table from data/churn.db, scores every customer with the
Phase 5 Logistic Regression model (the stronger performer per Phase 6's
AUC comparison), and writes one flat CSV with human-readable columns
(not one-hot encoded) plus two new columns: churn_probability and
predicted_churn. Power BI imports this directly via Get Data > Text/CSV.

Run:
    python scripts/generate_dashboard_data.py

Output:
    - outputs/dashboard_data.csv
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_model import load_features, preprocess

MODEL_PATH = Path("models/logistic_regression.joblib")
OUTPUT_CSV = Path("outputs/dashboard_data.csv")


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/train_model.py first (Phase 5).")
    return joblib.load(path)


def add_predictions(df: pd.DataFrame, model, threshold: float = 0.5) -> pd.DataFrame:
    """Score every row with the model and append readable prediction columns.

    Keeps the original human-readable columns (Contract, PaymentMethod,
    tenure_bucket, etc.) untouched -- one-hot encoding only happens
    internally for the model input, never in the output CSV.
    """
    df = df.copy()
    X, _y = preprocess(df)  # one-hot encoded, used only for scoring
    probabilities = model.predict_proba(X)[:, 1]

    df["churn_probability"] = probabilities.round(4)
    df["predicted_churn"] = ["Yes" if p >= threshold else "No" for p in probabilities]
    return df


def main() -> None:
    df = load_features()
    print(f"Loaded {len(df)} rows from features table")

    model = load_model()
    df_scored = add_predictions(df, model)
    print("Added churn_probability and predicted_churn columns")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_scored.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV} ({len(df_scored)} rows, {len(df_scored.columns)} columns)")
    print("\nOpen Power BI Desktop -> Get Data -> Text/CSV -> select this file to import.")


if __name__ == "__main__":
    main()