"""
tests/test_evaluate_models.py

Unit tests for scripts/evaluate_models.py. Trains tiny throwaway models
on synthetic data directly (rather than depending on data/churn.db or
models/*.joblib existing), so these run fast and standalone.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from evaluate_models import (
    get_classification_report,
    get_confusion_matrix,
    get_roc_data,
    get_top_features,
)


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n),
            "MonthlyCharges": rng.uniform(20, 120, n),
            "is_month_to_month": rng.integers(0, 2, n),
        }
    )
    # Make churn correlate loosely with tenure so metrics aren't degenerate
    y = pd.Series((X["tenure"] < 20).astype(int) ^ rng.integers(0, 2, n) // 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def toy_logistic_model(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def toy_rf_model(toy_data):
    X_train, X_test, y_train, y_test = toy_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model


def test_confusion_matrix_shape(toy_logistic_model, toy_data):
    _, X_test, _, y_test = toy_data
    cm = get_confusion_matrix(toy_logistic_model, X_test, y_test)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_test)


def test_classification_report_contains_both_classes(toy_logistic_model, toy_data):
    _, X_test, _, y_test = toy_data
    report = get_classification_report(toy_logistic_model, X_test, y_test)
    assert "No Churn" in report
    assert "Churn" in report


def test_roc_data_has_expected_keys_and_ranges(toy_logistic_model, toy_data):
    _, X_test, _, y_test = toy_data
    roc = get_roc_data(toy_logistic_model, X_test, y_test)
    assert set(roc.keys()) == {"fpr", "tpr", "auc"}
    assert 0.0 <= roc["auc"] <= 1.0
    assert len(roc["fpr"]) == len(roc["tpr"])


def test_top_features_logistic_regression_uses_coefficients(toy_logistic_model, toy_data):
    _, X_test, _, _ = toy_data
    top = get_top_features(toy_logistic_model, X_test.columns, top_n=3)
    assert len(top) == 3
    assert set(top.columns) == {"feature", "importance"}
    # sorted by absolute value, descending
    abs_vals = top["importance"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_top_features_random_forest_uses_importances(toy_rf_model, toy_data):
    _, X_test, _, _ = toy_data
    top = get_top_features(toy_rf_model, X_test.columns, top_n=3)
    assert len(top) == 3
    # random forest importances are always non-negative
    assert (top["importance"] >= 0).all()
    assert top["importance"].tolist() == sorted(top["importance"].tolist(), reverse=True)


def test_top_features_raises_for_unsupported_model():
    class FakeModel:
        pass

    with pytest.raises(ValueError):
        get_top_features(FakeModel(), ["a", "b"], top_n=2)


def test_top_features_unwraps_pipeline(toy_data):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = toy_data
    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(X_train, y_train)

    top = get_top_features(pipeline, X_test.columns, top_n=3)
    assert len(top) == 3
    assert set(top.columns) == {"feature", "importance"}