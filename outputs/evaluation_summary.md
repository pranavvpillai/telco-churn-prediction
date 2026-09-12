# Phase 6: Evaluation Summary

## Logistic Regression

AUC: 0.8421

```
              precision    recall  f1-score   support

    No Churn       0.84      0.90      0.87      1035
       Churn       0.65      0.52      0.58       374

    accuracy                           0.80      1409
   macro avg       0.75      0.71      0.72      1409
weighted avg       0.79      0.80      0.79      1409
```

Top features:

- tenure: -1.1624
- MonthlyCharges: -0.9019
- InternetService_Fiber optic: 0.7486
- Contract_Two year: -0.3581
- is_month_to_month: 0.3302
- StreamingTV_Yes: 0.2468
- StreamingMovies_Yes: 0.2465
- MultipleLines_Yes: 0.2323
- tenure_bucket_4yr+: 0.2295
- TotalCharges: 0.2123
- PaperlessBilling_Yes: 0.1865
- PaymentMethod_Electronic check: 0.1783
- OnlineSecurity_Yes: -0.1300
- tenure_bucket_1-2yr: -0.1235
- charge_to_tenure_ratio: 0.1056

## Random Forest

AUC: 0.8250

```
              precision    recall  f1-score   support

    No Churn       0.83      0.89      0.86      1035
       Churn       0.62      0.50      0.56       374

    accuracy                           0.79      1409
   macro avg       0.73      0.70      0.71      1409
weighted avg       0.78      0.79      0.78      1409
```

Top features:

- avg_charge_per_service: 0.1256
- TotalCharges: 0.1255
- tenure: 0.1112
- charge_to_tenure_ratio: 0.1085
- MonthlyCharges: 0.1046
- is_month_to_month: 0.0552
- PaymentMethod_Electronic check: 0.0325
- InternetService_Fiber optic: 0.0250
- gender_Male: 0.0217
- num_services: 0.0216
- PaperlessBilling_Yes: 0.0212
- Partner_Yes: 0.0163
- OnlineSecurity_Yes: 0.0152
- Contract_Two year: 0.0147
- Dependents_Yes: 0.0139
