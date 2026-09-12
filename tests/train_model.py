"""
Phase 5: Modeling
churn/scripts/train_model.py

Reads the `features` table built in Phase 4 (data/churn.db), preprocesses
it into a model-ready X/y, trains two models — a Logistic Regression
baseline (interpretable coefficients) and a Random Forest (stronger raw
predictive power) — and saves both to models/. Prints quick sanity-check
metrics; full evaluation (ROC curves, confusion matrices, feature
importance plots) is Phase 6.

Run:
    python scripts/train_model.py

Output:
    - models/logistic_regression.joblib
    - models/random_forest.joblib
    - printed accuracy / precision / recall / F1 / ROC-AUC for both models
"""

import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DB_PATH = Path("data/churn.db")
MODELS_DIR = Path("models")

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["customerID"]  # identifier, not a feature


def load_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load the Phase 4 `features` table."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Run scripts/load_data.py and "
            "scripts/feature_engineering.py first."
        )

    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM features", con)
    finally:
        con.close()
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split into X/y, encode target as 0/1, one-hot encode categoricals.

    Returns (X, y) where X is fully numeric and ready for sklearn.
    """
    df = df.copy()

    # Target: Yes/No -> 1/0
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == "Yes").astype(int)
    y = df[TARGET_COLUMN]

    # Drop identifier + target from the feature set
    cols_to_drop = [c for c in DROP_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # TotalCharges can arrive as a string with blanks for brand-new
    # (tenure == 0) customers in the raw Telco data; coerce defensively.
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

    # charge_to_tenure_ratio (Phase 4) is intentionally NaN for tenure == 0
    # customers (division by zero is undefined). Models can't accept NaN,
    # so treat "no ratio yet" as 0 here.
    if "charge_to_tenure_ratio" in X.columns:
        X["charge_to_tenure_ratio"] = X["charge_to_tenure_ratio"].fillna(0)

    # One-hot encode every remaining categorical/object column
    # (this includes tenure_bucket from Phase 4).
    X = pd.get_dummies(X, drop_first=True)

    # Safety net: catch any other stray NaNs (e.g. from unexpected nulls
    # in future data) rather than letting sklearn crash on them.
    X = X.fillna(0)

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Stratified train/test split so churn rate is preserved in both sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_quick(model, X_test, y_test) -> dict:
    """Quick sanity-check metrics. Full evaluation is Phase 6."""
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probs)
    return metrics


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key:10s}: {value:.4f}")


def main() -> None:
    df = load_features()
    print(f"Loaded {len(df)} rows from features table")

    X, y = preprocess(df)
    print(f"Preprocessed to {X.shape[1]} features (after one-hot encoding)")

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    log_reg = train_logistic_regression(X_train, y_train)
    log_reg_metrics = evaluate_quick(log_reg, X_test, y_test)
    print_metrics("Logistic Regression (baseline)", log_reg_metrics)
    save_model(log_reg, MODELS_DIR / "logistic_regression.joblib")

    rf = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_quick(rf, X_test, y_test)
    print_metrics("Random Forest", rf_metrics)
    save_model(rf, MODELS_DIR / "random_forest.joblib")

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()