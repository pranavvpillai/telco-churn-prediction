"""
tests/test_train_model.py

Unit tests for scripts/train_model.py. Uses a small synthetic dataset
shaped like the `features` table, so these run fast and don't depend
on data/churn.db existing yet.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_model import (
    evaluate_quick,
    preprocess,
    save_model,
    split_data,
    train_logistic_regression,
    train_random_forest,
)


@pytest.fixture
def sample_features_df():
    rng = np.random.default_rng(0)
    n = 100
    return pd.DataFrame(
        {
            "customerID": [str(i) for i in range(n)],
            "tenure": rng.integers(0, 72, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "TotalCharges": rng.uniform(0, 8000, n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "tenure_bucket": rng.choice(["0-1yr", "1-2yr", "2-4yr", "4yr+"], n),
            "num_services": rng.integers(0, 6, n),
            "avg_charge_per_service": rng.uniform(5, 50, n),
            "charge_to_tenure_ratio": rng.uniform(0, 200, n),
            "is_month_to_month": rng.integers(0, 2, n),
            "has_internet": rng.integers(0, 2, n),
            "senior_and_alone": rng.integers(0, 2, n),
            "Churn": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        }
    )


def test_preprocess_encodes_target_as_binary(sample_features_df):
    X, y = preprocess(sample_features_df)
    assert set(y.unique()).issubset({0, 1})


def test_preprocess_drops_identifier_and_target(sample_features_df):
    X, y = preprocess(sample_features_df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_preprocess_produces_fully_numeric_frame(sample_features_df):
    X, y = preprocess(sample_features_df)
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_preprocess_handles_nan_charge_to_tenure_ratio(sample_features_df):
    # tenure == 0 customers get NaN charge_to_tenure_ratio in Phase 4 by
    # design (division by zero). preprocess() must not pass NaN to sklearn.
    df = sample_features_df.copy()
    df.loc[0, "charge_to_tenure_ratio"] = float("nan")
    X, y = preprocess(df)
    assert not X.isna().any().any()


def test_preprocess_one_hot_encodes_categoricals(sample_features_df):
    X, y = preprocess(sample_features_df)
    # Contract had 3 categories -> drop_first means 2 dummy columns exist
    contract_cols = [c for c in X.columns if c.startswith("Contract_")]
    assert len(contract_cols) == 2


def test_split_data_preserves_row_count(sample_features_df):
    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_split_data_is_stratified(sample_features_df):
    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    # churn rate in test set should be roughly close to overall churn rate
    overall_rate = y.mean()
    test_rate = y_test.mean()
    assert abs(overall_rate - test_rate) < 0.15


def test_logistic_regression_trains_and_predicts(sample_features_df):
    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_logistic_regression(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_random_forest_trains_and_predicts(sample_features_df):
    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})


def test_evaluate_quick_returns_expected_keys(sample_features_df):
    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_quick(model, X_test, y_test)
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert expected_keys.issubset(metrics.keys())
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_save_model_roundtrip(sample_features_df, tmp_path):
    import joblib

    X, y = preprocess(sample_features_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_logistic_regression(X_train, y_train)

    save_path = tmp_path / "model.joblib"
    save_model(model, save_path)
    assert save_path.exists()

    loaded = joblib.load(save_path)
    assert (loaded.predict(X_test) == model.predict(X_test)).all()