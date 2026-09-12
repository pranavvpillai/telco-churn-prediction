"""Tests for scripts/load_data.py."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from load_data import clean_raw, build_tables


@pytest.fixture
def raw_df():
    return pd.DataFrame({
        "customerID": ["0001-AAA", "0002-BBB", "0003-CCC"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [0, 12, 24],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 20.15],
        # blank string is how the real dataset encodes a brand-new customer's
        # TotalCharges (tenure=0) -- this is the known real-world quirk.
        "TotalCharges": ["", "684.5", "483.6"],
        "Churn": ["No", "Yes", "No"],
    })


class TestCleanRaw:
    def test_total_charges_blank_becomes_nan(self, raw_df):
        cleaned = clean_raw(raw_df)
        assert pd.isna(cleaned.loc[0, "TotalCharges"])

    def test_total_charges_valid_values_parsed_as_numeric(self, raw_df):
        cleaned = clean_raw(raw_df)
        assert cleaned.loc[1, "TotalCharges"] == 684.5
        assert cleaned["TotalCharges"].dtype.kind == "f"

    def test_does_not_mutate_original(self, raw_df):
        original_dtype = raw_df["TotalCharges"].dtype
        clean_raw(raw_df)
        assert raw_df["TotalCharges"].dtype == original_dtype


class TestBuildTables:
    def test_splits_into_three_tables(self, raw_df):
        df = clean_raw(raw_df)
        customers, services, billing = build_tables(df)
        assert len(customers) == len(services) == len(billing) == 3

    def test_customerID_present_in_all_tables_for_joining(self, raw_df):
        df = clean_raw(raw_df)
        customers, services, billing = build_tables(df)
        assert "customerID" in customers.columns
        assert "customerID" in services.columns
        assert "customerID" in billing.columns

    def test_no_data_loss_across_split(self, raw_df):
        df = clean_raw(raw_df)
        customers, services, billing = build_tables(df)
        # every original column should land in exactly one of the three tables
        # (customerID is the shared join key, so it appears in all three)
        all_cols = set(customers.columns) | set(services.columns) | set(billing.columns)
        assert all_cols == set(df.columns)

    def test_churn_column_only_in_customers(self, raw_df):
        df = clean_raw(raw_df)
        customers, services, billing = build_tables(df)
        assert "Churn" in customers.columns
        assert "Churn" not in services.columns
        assert "Churn" not in billing.columns

    def test_billing_values_preserved(self, raw_df):
        df = clean_raw(raw_df)
        customers, services, billing = build_tables(df)
        assert billing.loc[1, "MonthlyCharges"] == 56.95
