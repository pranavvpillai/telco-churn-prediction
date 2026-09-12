"""Statistical testing: which factors actually drive churn, vs. which just
look interesting in a raw percentage.

Chi-square test of independence for categorical features (contract type,
internet service, payment method, etc.) vs. Churn, with Cramer's V as an
effect-size measure. Welch's t-test for numeric features (tenure,
MonthlyCharges, TotalCharges), with Cohen's d as an effect-size measure.

With a large sample (n=7043), statistical significance (p<0.05) is easy
to achieve even for practically tiny effects -- p-values shrink with
sample size and only tell you a relationship is real, not that it's
large. Effect size is what tells you whether a significant feature
actually matters for the business.

Usage:
    python scripts/statistical_tests.py
"""
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "churn.db"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "statistical_significance_summary.csv"

ALPHA = 0.05  # standard significance threshold

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "Contract",
    "PaperlessBilling", "PaymentMethod",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_full_dataset() -> pd.DataFrame:
    """Join all three tables back into one analysis-ready dataframe."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT c.*, s.PhoneService, s.MultipleLines, s.InternetService,
               s.OnlineSecurity, s.OnlineBackup, s.DeviceProtection,
               s.TechSupport, s.StreamingTV, s.StreamingMovies,
               b.MonthlyCharges, b.TotalCharges
        FROM customers c
        JOIN services s ON c.customerID = s.customerID
        JOIN billing b ON c.customerID = b.customerID
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def cramers_v(contingency: pd.DataFrame) -> float:
    """Effect size for a chi-square test. 0 = no association, 1 = perfect.
    Rule of thumb: <0.1 negligible, 0.1-0.3 small, 0.3-0.5 moderate, >0.5 large.
    """
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    return float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else 0.0


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Effect size for a t-test (standardized mean difference).
    Rule of thumb: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
    """
    pooled_std = np.sqrt(((a.std() ** 2) + (b.std() ** 2)) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def effect_label(value: float) -> str:
    value = abs(value)
    if value < 0.1:
        return "negligible"
    if value < 0.3:
        return "small"
    if value < 0.5:
        return "moderate"
    return "large"


def chi_square_test(df: pd.DataFrame, feature: str, target: str = "Churn") -> dict:
    """Chi-square test of independence between a categorical feature and churn."""
    contingency = pd.crosstab(df[feature], df[target])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    v = cramers_v(contingency)
    return {
        "feature": feature,
        "test": "chi-square",
        "statistic": round(chi2, 3),
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "effect_size": round(v, 3),
        "effect_size_metric": "Cramer's V",
        "effect_magnitude": effect_label(v),
    }


def t_test(df: pd.DataFrame, feature: str, target: str = "Churn") -> dict:
    """Welch's t-test comparing a numeric feature between churned and retained customers."""
    churned = df.loc[df[target] == "Yes", feature].dropna()
    retained = df.loc[df[target] == "No", feature].dropna()
    t_stat, p_value = stats.ttest_ind(churned, retained, equal_var=False)  # Welch's
    d = cohens_d(churned, retained)
    return {
        "feature": feature,
        "test": "welch_t_test",
        "statistic": round(t_stat, 3),
        "p_value": p_value,
        "significant": p_value < ALPHA,
        "effect_size": round(d, 3),
        "effect_size_metric": "Cohen's d",
        "effect_magnitude": effect_label(d),
        "churned_mean": round(churned.mean(), 2),
        "retained_mean": round(retained.mean(), 2),
    }


def run_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for feature in CATEGORICAL_FEATURES:
        results.append(chi_square_test(df, feature))
    for feature in NUMERIC_FEATURES:
        results.append(t_test(df, feature))

    results_df = pd.DataFrame(results)
    # rank by effect size, not p-value -- p-value only says "real", effect size says "how much"
    results_df = results_df.sort_values("effect_size", ascending=False)
    return results_df


def main():
    df = load_full_dataset()
    results_df = run_all_tests(df)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    sig = results_df[results_df["significant"]]
    not_sig = results_df[~results_df["significant"]]

    print(f"Tested {len(results_df)} features against Churn (alpha = {ALPHA})")
    print(f"Ranked by EFFECT SIZE (not p-value) -- with n={len(df)}, p-value alone "
          f"is a weak signal of practical importance.\n")

    print(f"STATISTICALLY SIGNIFICANT ({len(sig)} features), ranked by effect magnitude:")
    for _, row in sig.iterrows():
        print(f"  {row['feature']:<20} {row['effect_size_metric']:<12}={row['effect_size']:<6} "
              f"({row['effect_magnitude']:<10}) p={row['p_value']:.2e}")

    print(f"\nNOT statistically significant ({len(not_sig)} features):")
    for _, row in not_sig.iterrows():
        print(f"  {row['feature']:<20} p={row['p_value']:.3f}")

    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()