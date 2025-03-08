import os
import joblib
import numpy as np
import tensorflow as tf
import logging
from flask import Flask, request, jsonify
from income_forecasting_service import forecast_income_expense
from keras.saving import register_keras_serializable

# ✅ Configure Logging
logging.basicConfig(level=logging.INFO)

# ✅ Register custom MSE function (for LSTM model compatibility)
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# ✅ Define Paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")

risk_model_path = os.path.join(models_dir, "risk_model.pkl")
lstm_model_path = os.path.join(models_dir, "lstm_model.h5")
fraud_model_path = os.path.join(models_dir, "fraud_detection_model.pkl")

# ✅ Ensure models exist before loading
for path, name in [(risk_model_path, "Risk Model"), (lstm_model_path, "LSTM Model"), (fraud_model_path, "Fraud Detection Model")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ ERROR: {name} file is missing!")

# ✅ Load Models
logging.info("✅ Loading models...")
risk_model = joblib.load(risk_model_path)
lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects={'mse': mse})
fraud_model = joblib.load(fraud_model_path)
logging.info("✅ All models loaded successfully!")

# ✅ Initialize Flask App
app = Flask(__name__)

# 📌 **Risk Analysis API**
@app.route('/analyze_risk', methods=['POST'])
def analyze_risk():
    """
    Analyzes the risk level based on customer transaction data.
    """
    data = request.get_json()
    if not data or "transaction_data" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    transactions = data["transaction_data"]

    features = np.array([
        transactions.get("salary_deposits", 0),
        transactions.get("total_expenses", 0),
        transactions.get("overdraft_count", 0),
        transactions.get("expense_ratio", 0.0),
        transactions.get("credit_score", 600)
    ]).reshape(1, -1)

    prediction = risk_model.predict(features)
    risk_map = {0: "low", 1: "medium", 2: "high"}

    return jsonify({"risk_level": risk_map.get(prediction[0], "unknown")})

# 📌 **Income Forecasting API**
@app.route('/forecast_income', methods=['POST'])
def forecast_income():
    """
    Forecasts future income and expenses using the LSTM model.
    """
    data = request.get_json()
    if not data or "history" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    try:
        result = forecast_income_expense(data["history"])
        return jsonify(result)
    except Exception as e:
        logging.error(f"❌ Error in income forecasting: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# 📌 **Fraud Detection Function**
def detect_fraud(transaction):
    """
    Detects whether a transaction is fraudulent.

    :param transaction: dict containing transaction details
    :return: dict with fraud risk or error message
    """
    try:
        # Extract expected feature set (ensure all 15 features exist)
        expected_features = [
            "amount", "transaction_type", "merchant_category", "time_of_day",
            "feature_5", "feature_6", "feature_7", "feature_8", "feature_9",
            "feature_10", "feature_11", "feature_12", "feature_13", "feature_14", "feature_15"
        ]

        # Convert transaction data into a feature vector
        transaction_vector = np.array([[transaction.get(f, 0) for f in expected_features]])

        # Ensure correct input size
        if transaction_vector.shape[1] != fraud_model.n_features_in_:
            return {"error": f"Expected {fraud_model.n_features_in_} features, got {transaction_vector.shape[1]}"}

        prediction = fraud_model.predict(transaction_vector)
        return {"fraud_risk": "high" if prediction[0] == -1 else "low"}

    except Exception as e:
        logging.error(f"❌ Error in fraud detection: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

# 📌 **Fraud Detection API**
@app.route('/detect_fraud', methods=['POST'])
def detect_fraud_api():
    """
    Flask API endpoint for fraud detection.
    """
    data = request.get_json()
    if not data or "transaction" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    result = detect_fraud(data["transaction"])
    return jsonify(result)

# ✅ Run the Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
