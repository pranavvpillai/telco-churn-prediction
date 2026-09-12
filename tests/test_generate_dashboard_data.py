"""
tests/test_generate_dashboard_data.py

Unit tests for scripts/generate_dashboard_data.py. Trains a tiny throwaway
logistic regression on synthetic data (matching the shape of the real
`features` table) rather than depending on models/logistic_regression.joblib
existing, so these run fast and standalone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from generate_dashboard_data import add_predictions
from train_model import preprocess


@pytest.fixture
def sample_features_df():
    rng = np.random.default_rng(0)
    n = 50
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


@pytest.fixture
def toy_model(sample_features_df):
    X, y = preprocess(sample_features_df)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def test_add_predictions_preserves_original_columns(sample_features_df, toy_model):
    original_cols = set(sample_features_df.columns)
    result = add_predictions(sample_features_df, toy_model)
    assert original_cols.issubset(set(result.columns))


def test_add_predictions_does_not_one_hot_encode_output(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model)
    # Contract should remain readable strings, not dummy columns
    assert "Contract" in result.columns
    assert set(result["Contract"].unique()).issubset({"Month-to-month", "One year", "Two year"})
    assert not any(c.startswith("Contract_") for c in result.columns)


def test_add_predictions_adds_expected_new_columns(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model)
    assert "churn_probability" in result.columns
    assert "predicted_churn" in result.columns


def test_churn_probability_is_between_zero_and_one(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model)
    assert (result["churn_probability"] >= 0).all()
    assert (result["churn_probability"] <= 1).all()


def test_predicted_churn_is_yes_or_no(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model)
    assert set(result["predicted_churn"].unique()).issubset({"Yes", "No"})


def test_predicted_churn_matches_threshold(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model, threshold=0.5)
    for _, row in result.iterrows():
        expected = "Yes" if row["churn_probability"] >= 0.5 else "No"
        assert row["predicted_churn"] == expected


def test_row_count_unchanged(sample_features_df, toy_model):
    result = add_predictions(sample_features_df, toy_model)
    assert len(result) == len(sample_features_df)