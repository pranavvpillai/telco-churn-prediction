# Telco Customer Churn — Analysis & Prediction Pipeline

An end-to-end churn analysis project on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers): SQL-based data modeling, statistical hypothesis testing with effect sizes, feature engineering, two classification models, full evaluation, and an interactive Power BI dashboard.

## Project Structure

```
data_science_project/
├── data/
│   ├── Telco-Customer-Churn.csv       # raw Kaggle dataset
│   └── churn.db                       # SQLite: customers, services, billing, features tables
├── sql/
│   └── queries.sql                    # joins, CTEs, window functions, data-quality audit
├── scripts/
│   ├── load_data.py                   # Phase 1: CSV -> normalized SQLite tables
│   ├── run_queries.py                 # Phase 2: runs sql/queries.sql
│   ├── statistical_tests.py           # Phase 3: chi-square / t-tests + effect sizes
│   ├── feature_engineering.py         # Phase 4: derived features -> features table
│   ├── train_model.py                 # Phase 5: preprocessing + model training
│   ├── evaluate_models.py             # Phase 6: confusion matrices, ROC, feature importance
│   └── generate_dashboard_data.py     # Phase 7: scored CSV export for Power BI
├── tests/                             # pytest suite, 54 tests across all phases
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
├── outputs/
│   ├── statistical_significance_summary.csv
│   ├── engineered_features.csv
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   ├── feature_importance_logistic_regression.png
│   ├── feature_importance_random_forest.png
│   ├── evaluation_summary.md
│   └── dashboard_data.csv             # Power BI data source
├── dashboard/                         # Power BI .pbix file
└── requirements.txt
```

## How to Run

```powershell
pip install -r requirements.txt
python scripts/load_data.py
python scripts/run_queries.py
python scripts/statistical_tests.py
python scripts/feature_engineering.py
python scripts/train_model.py
python scripts/evaluate_models.py
python scripts/generate_dashboard_data.py
pytest
```

`pytest` should report **54 passed**. Open `outputs/dashboard_data.csv` in Power BI Desktop (Get Data → Text/CSV) to rebuild the dashboard.

## Key Findings

### Statistical significance vs. practical significance

19 features were tested against `Churn` (chi-square for categoricals, Welch's t-test for numerics). 17 came back statistically significant at α = 0.05 — but with n = 7,043, statistical significance is cheap. Effect sizes (Cramér's V, Cohen's d) tell the real story:

| Feature | Effect size | Magnitude |
|---|---|---|
| tenure | Cohen's d = -0.89 | **Large** — the single strongest driver |
| Contract | Cramér's V = 0.41 | Moderate |
| MonthlyCharges | Cohen's d = 0.47 | Moderate |
| OnlineSecurity | Cramér's V = 0.35 | Moderate |
| TechSupport | Cramér's V = 0.34 | Moderate |
| InternetService | Cramér's V = 0.32 | Moderate |
| PaymentMethod | Cramér's V = 0.30 | Moderate |
| MultipleLines | Cramér's V = 0.04 | **Negligible** — significant but practically meaningless |

`MultipleLines` is the clearest illustration of the distinction: p = 3.46e-03 says "this relationship is real," while Cramér's V = 0.04 says "it barely matters in practice." `PhoneService` and `gender` were not statistically significant at all.

### Feature engineering

Seven derived features were added on top of the raw columns: `tenure_bucket`, `num_services`, `avg_charge_per_service`, `charge_to_tenure_ratio`, `is_month_to_month`, `has_internet`, and `senior_and_alone`.

### Modeling & evaluation

Two models were trained on an 80/20 stratified split (Logistic Regression wrapped in a `StandardScaler` pipeline; Random Forest, 200 trees):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.799 | 0.65 | 0.52 | 0.58 | **0.842** |
| Random Forest | 0.786 | 0.62 | 0.50 | 0.56 | 0.825 |

Logistic Regression edged out Random Forest on every metric and was selected as the model powering the dashboard. Recall on the Churn class (~0.52) is the main limitation — the model still misses roughly half of actual churners, which is the natural next thing to improve (class weighting, threshold tuning, or SMOTE).

**Calibration note:** an early version of the Logistic Regression model, fit without feature scaling, produced badly miscalibrated probabilities (average predicted churn probability of 0.66 against an actual base rate of 0.265) due to the optimizer failing to converge on unscaled features (`MonthlyCharges`/`TotalCharges` alongside 0/1 dummy columns). Adding `StandardScaler` to the pipeline fixed this — average predicted probability now matches the actual churn rate almost exactly (0.2657 vs. 0.2654).

## Dashboard

Built in Power BI from `outputs/dashboard_data.csv` (every customer, human-readable columns, plus `churn_probability` and `predicted_churn` from the Logistic Regression model). Includes:

- KPI cards: average churn probability, count of predicted churners
- Churn risk by Contract type (confirms Month-to-month customers dominate the at-risk group)
- A sortable "Top At-Risk Customers" table (customerID, tenure, MonthlyCharges, Contract, churn_probability)
- Interactive slicers: tenure bucket, internet service type, payment method


![Churn Risk Dashboard](outputs/dashboard_screenshot.png)


## Testing

Every phase has a corresponding pytest suite (`tests/`), 54 tests total, covering data loading, SQL correctness, statistical function correctness (Cramér's V, Cohen's d, effect labeling), feature engineering edge cases (NaN handling for zero-tenure customers), model preprocessing, and dashboard export logic.
