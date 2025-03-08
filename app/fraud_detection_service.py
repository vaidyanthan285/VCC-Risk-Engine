import os
import joblib
import numpy as np
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")
fraud_model_path = os.path.join(models_dir, "fraud_detection_model.pkl")

# Load Model
if not os.path.exists(fraud_model_path):
    raise FileNotFoundError("❌ ERROR: Fraud model file is missing!")

fraud_model = joblib.load(fraud_model_path)
logging.info("✅ Fraud detection model loaded successfully.")

# Initialize Flask App
app = Flask(__name__)

# ✅ Define the detect_fraud function before using it in the API route
def detect_fraud(transaction):
    """
    Detects whether a transaction is fraudulent.

    :param transaction: dict containing transaction details
    :return: dict with fraud risk or error message
    """
    try:
        # Convert transaction data into a feature vector
        transaction_vector = np.array([[
            transaction.get("amount", 0),
            transaction.get("transaction_type", 0),
            transaction.get("merchant_category", 0),
            transaction.get("time_of_day", 0)
        ]])

        # Ensure the input matches model expectations
        if transaction_vector.shape[1] != fraud_model.n_features_in_:
            return {"error": f"Expected {fraud_model.n_features_in_} features, got {transaction_vector.shape[1]}"}

        prediction = fraud_model.predict(transaction_vector)
        return {"fraud_risk": "high" if prediction[0] == -1 else "low"}

    except Exception as e:
        logging.error(f"❌ Error in fraud detection: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}