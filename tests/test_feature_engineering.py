"""
tests/test_feature_engineering.py

Unit tests for scripts/feature_engineering.py. Uses small hand-built
DataFrames rather than the real churn.db, so these run fast and don't
depend on the dataset being loaded yet.
"""

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from feature_engineering import (
    add_avg_charge_per_service,
    add_charge_to_tenure_ratio,
    add_has_internet,
    add_is_month_to_month,
    add_senior_and_alone,
    add_service_count,
    add_tenure_bucket,
    engineer_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "customerID": ["1", "2", "3"],
            "tenure": [0, 15, 50],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "MonthlyCharges": [70.0, 50.0, 90.0],
            "TotalCharges": [0.0, 750.0, 4500.0],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "SeniorCitizen": [1, 0, 1],
            "Partner": ["No", "Yes", "No"],
            "Dependents": ["No", "No", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "No", "No internet service"],
            "StreamingMovies": ["No", "No", "No internet service"],
        }
    )


def test_tenure_bucket_assigns_expected_ranges(sample_df):
    df = add_tenure_bucket(sample_df)
    assert df.loc[0, "tenure_bucket"] == "0-1yr"
    assert df.loc[1, "tenure_bucket"] == "1-2yr"
    assert df.loc[2, "tenure_bucket"] == "4yr+"


def test_service_count_counts_yes_values(sample_df):
    df = add_service_count(sample_df)
    assert df.loc[0, "num_services"] == 2  # OnlineSecurity + OnlineBackup
    assert df.loc[1, "num_services"] == 1  # TechSupport
    assert df.loc[2, "num_services"] == 0  # no internet -> no add-ons


def test_avg_charge_per_service_divides_correctly(sample_df):
    df = add_service_count(sample_df)
    df = add_avg_charge_per_service(df)
    # row 0: 70 / (2 + 1) = 23.33...
    assert df.loc[0, "avg_charge_per_service"] == pytest.approx(70.0 / 3)


def test_charge_to_tenure_ratio_handles_zero_tenure(sample_df):
    df = add_charge_to_tenure_ratio(sample_df)
    assert math.isnan(df.loc[0, "charge_to_tenure_ratio"])
    assert df.loc[1, "charge_to_tenure_ratio"] == pytest.approx(750.0 / 15)


def test_is_month_to_month_flags_correctly(sample_df):
    df = add_is_month_to_month(sample_df)
    assert df.loc[0, "is_month_to_month"] == 1
    assert df.loc[1, "is_month_to_month"] == 0
    assert df.loc[2, "is_month_to_month"] == 0


def test_has_internet_flags_correctly(sample_df):
    df = add_has_internet(sample_df)
    assert df.loc[0, "has_internet"] == 1
    assert df.loc[1, "has_internet"] == 1
    assert df.loc[2, "has_internet"] == 0


def test_senior_and_alone_flags_correctly(sample_df):
    df = add_senior_and_alone(sample_df)
    assert df.loc[0, "senior_and_alone"] == 1  # senior, no partner, no dependents
    assert df.loc[1, "senior_and_alone"] == 0  # not senior
    assert df.loc[2, "senior_and_alone"] == 1  # senior, no partner, no dependents


def test_engineer_features_adds_all_columns(sample_df):
    df = engineer_features(sample_df)
    expected_new_cols = {
        "tenure_bucket",
        "num_services",
        "avg_charge_per_service",
        "charge_to_tenure_ratio",
        "is_month_to_month",
        "has_internet",
        "senior_and_alone",
    }
    assert expected_new_cols.issubset(set(df.columns))
    assert len(df) == 3  # no rows dropped