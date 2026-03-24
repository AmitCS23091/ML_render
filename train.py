# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Create simple dataset
data = {
    "income": [20000, 50000, 80000, 30000, 90000, 40000, 70000, 100000],
    "credit_score": [600, 750, 800, 650, 850, 700, 780, 900],
    "loan_amount": [100000, 200000, 300000, 150000, 400000, 180000, 250000, 500000],
    "employment": [0, 1, 1, 0, 1, 1, 1, 1],
    "approved": [0, 1, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[["income", "credit_score", "loan_amount", "employment"]]
y = df["approved"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved successfully!")