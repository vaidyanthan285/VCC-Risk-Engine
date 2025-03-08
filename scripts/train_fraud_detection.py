import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

# Define correct paths dynamically
base_dir = os.getcwd()  # Get current working directory
data_path = os.path.join(base_dir, 'data', 'customer_transactions.csv')
model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)  # Ensure models directory exists

# Load Data
df = pd.read_csv(data_path)

# Select relevant features
features = ['amount', 'transaction_type', 'merchant_category', 'time_of_day']

df = df[features]

# Encode categorical variables
df = pd.get_dummies(df)

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(df)

# Save Model
model_path = os.path.join(model_dir, 'fraud_detection_model.pkl')
joblib.dump(model, model_path)

print(f"✅ Fraud Detection Model Trained and Saved at {model_path}")
