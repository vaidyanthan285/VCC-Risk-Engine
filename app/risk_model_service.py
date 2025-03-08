import os
import joblib
import numpy as np

# Define paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")
risk_model_path = os.path.join(models_dir, "risk_model.pkl")

# Load Model
if not os.path.exists(risk_model_path):
    raise FileNotFoundError("❌ ERROR: Risk model file is missing!")

risk_model = joblib.load(risk_model_path)

def predict_risk(transaction_data):
    """
    Predicts risk level based on transaction data.

    :param transaction_data: dict containing customer transaction details
    :return: str (low, medium, or high risk)
    """
    features = np.array([
        transaction_data.get("salary_deposits", 0),
        transaction_data.get("total_expenses", 0),
        transaction_data.get("overdraft_count", 0),
        transaction_data.get("expense_ratio", 0.0),
        transaction_data.get("credit_score", 600)
    ]).reshape(1, -1)

    prediction = risk_model.predict(features)
    risk_map = {0: "low", 1: "medium", 2: "high"}

    return risk_map[prediction[0]]
