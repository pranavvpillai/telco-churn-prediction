"""Tests for the effect-size functions in statistical_tests.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from statistical_tests import cramers_v, cohens_d, effect_label


class TestCramersV:
    def test_independent_variables_give_near_zero(self):
        # rows and columns generated independently -- no real association
        rows = ["A", "A", "B", "B"] * 50
        cols = ["X", "Y"] * 100
        contingency = pd.crosstab(pd.Series(rows), pd.Series(cols))
        v = cramers_v(contingency)
        assert v < 0.05

    def test_perfect_association_gives_near_one(self):
        contingency = pd.DataFrame([[100, 0], [0, 100]])
        v = cramers_v(contingency)
        assert v > 0.9

    def test_result_is_non_negative(self):
        contingency = pd.DataFrame([[30, 20], [15, 35]])
        assert cramers_v(contingency) >= 0

    def test_single_column_returns_zero_not_error(self):
        # min_dim = 0 case -- must not divide by zero
        contingency = pd.DataFrame([[10], [20]])
        assert cramers_v(contingency) == 0.0


class TestCohensD:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(50, 5, 1000))
        b = pd.Series(rng.normal(50, 5, 1000))
        d = cohens_d(a, b)
        assert abs(d) < 0.1

    def test_known_separation_matches_expected_d(self):
        # two std devs apart, same std dev -> d approx 2.0
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(60, 5, 2000))
        b = pd.Series(rng.normal(50, 5, 2000))
        d = cohens_d(a, b)
        assert 1.8 < d < 2.2

    def test_direction_reflects_which_group_is_higher(self):
        higher = pd.Series([10.0] * 50)
        lower = pd.Series([5.0] * 50)
        # zero variance is a degenerate edge case (division by zero pooled_std);
        # use near-constant values instead to keep the direction check meaningful
        higher = higher + pd.Series(np.random.default_rng(1).normal(0, 0.01, 50))
        lower = lower + pd.Series(np.random.default_rng(2).normal(0, 0.01, 50))
        assert cohens_d(higher, lower) > 0
        assert cohens_d(lower, higher) < 0

    def test_zero_variance_does_not_raise(self):
        a = pd.Series([10.0] * 20)
        b = pd.Series([10.0] * 20)
        assert cohens_d(a, b) == 0.0


class TestEffectLabel:
    @pytest.mark.parametrize("value,expected", [
        (0.05, "negligible"),
        (0.15, "small"),
        (0.35, "moderate"),
        (0.6, "large"),
        (-0.6, "large"),  # magnitude only -- sign shouldn't matter
    ])
    def test_thresholds(self, value, expected):
        assert effect_label(value) == expected