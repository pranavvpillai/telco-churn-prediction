"""Generate a small synthetic sample matching the real IBM Telco Customer Churn
schema, purely so the pipeline can be developed and tested before the real
7,043-row dataset is dropped in. NOT used for the actual analysis.
"""
import random
import csv

random.seed(42)

genders = ["Male", "Female"]
yes_no = ["Yes", "No"]
internet_options = ["DSL", "Fiber optic", "No"]
contract_options = ["Month-to-month", "One year", "Two year"]
payment_options = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]

rows = []
for i in range(300):
    internet = random.choice(internet_options)
    has_internet = internet != "No"
    tenure = random.randint(0, 72)
    monthly = round(random.uniform(18.0, 120.0), 2)
    total = round(monthly * max(tenure, 1) * random.uniform(0.9, 1.05), 2)
    churn = random.choices(yes_no, weights=[27, 73])[0]  # realistic ~27% churn rate
    rows.append({
        "customerID": f"{random.randint(1000,9999)}-{''.join(random.choices('ABCDEFGH', k=5))}",
        "gender": random.choice(genders),
        "SeniorCitizen": random.choices([0, 1], weights=[84, 16])[0],
        "Partner": random.choice(yes_no),
        "Dependents": random.choice(yes_no),
        "tenure": tenure,
        "PhoneService": random.choices(yes_no, weights=[90, 10])[0],
        "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
        "InternetService": internet,
        "OnlineSecurity": random.choice(yes_no) if has_internet else "No internet service",
        "OnlineBackup": random.choice(yes_no) if has_internet else "No internet service",
        "DeviceProtection": random.choice(yes_no) if has_internet else "No internet service",
        "TechSupport": random.choice(yes_no) if has_internet else "No internet service",
        "StreamingTV": random.choice(yes_no) if has_internet else "No internet service",
        "StreamingMovies": random.choice(yes_no) if has_internet else "No internet service",
        "Contract": random.choice(contract_options),
        "PaperlessBilling": random.choice(yes_no),
        "PaymentMethod": random.choice(payment_options),
        "MonthlyCharges": monthly,
        "TotalCharges": total if tenure > 0 else "",  # real dataset has blanks for tenure=0
        "Churn": churn,
    })

with open("/home/claude/churn/data/Telco-Customer-Churn-SAMPLE.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} synthetic sample rows")
