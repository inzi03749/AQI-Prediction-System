# ===============================
# AQI Model Training Script
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import joblib
import os

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("final_dataset.csv")
df = df.fillna(df.mean())

#Histogram Plot
plt.figure(figsize=(8,5))
plt.hist(df["AQI"], bins=20)
plt.title("Distribution of AQI")
plt.xlabel("AQI")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

target = "AQI"
X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Performance")
print("----------------------")
print("R2 Score:", round(r2 * 100, 2), "%")
print("RMSE:", round(rmse, 2))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")

print("\nModel saved successfully in models/xgb_model.pkl")

# -----------------------------
# Plot Actual vs Predicted
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Actual vs Predicted AQI")
plt.show()