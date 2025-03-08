import os
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from keras.saving import register_keras_serializable

# Register custom mse function (if needed)
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")

# Ensure models exist
risk_model_path = os.path.join(models_dir, "risk_model.pkl")
lstm_model_path = os.path.join(models_dir, "lstm_model.h5")
fraud_model_path = os.path.join(models_dir, "fraud_detection_model.pkl")

if not os.path.exists(risk_model_path) or not os.path.exists(lstm_model_path) or not os.path.exists(fraud_model_path):
    raise FileNotFoundError("❌ ERROR: One or more model files are missing!")

# Load Models
risk_model = joblib.load(risk_model_path)
lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects={'mse': mse})  # Add custom_objects
fraud_model = joblib.load(fraud_model_path)

# Initialize Flask App
app = Flask(__name__)

@app.route('/analyze_risk', methods=['POST'])
def analyze_risk():
    data = request.get_json()
    
    if not data or "transaction_data" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    transactions = data["transaction_data"]

    features = [
        transactions.get("salary_deposits", 0),
        transactions.get("total_expenses", 0),
        transactions.get("overdraft_count", 0),
        transactions.get("expense_ratio", 0.0),
        transactions.get("credit_score", 600)
    ]

    prediction = risk_model.predict([features])
    risk_map = {0: "low", 1: "medium", 2: "high"}

    return jsonify({"risk_level": risk_map[prediction[0]]})

@app.route('/forecast_income', methods=['POST'])
def forecast_income():
    data = request.get_json()

    if not data or "history" not in data:
        return jsonify({"error": "Invalid request format"}), 400

    try:
        input_data = np.array([data["history"]]).reshape(1, len(data["history"]), 2)
        prediction = lstm_model.predict(input_data)[0].tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "predicted_salary": prediction[0],
        "predicted_expenses": prediction[1]
    })

# ✅ Define the missing detect_fraud function
def detect_fraud(transaction):
    """
    Detects whether a transaction is fraudulent.
    
    :param transaction: dict containing transaction details
    :return: dict with fraud risk or error message
    """
    try:
        # Extract all 15 expected features (replace placeholders with actual feature names)
        transaction_vector = np.array([[  
            transaction.get("amount", 0),
            transaction.get("transaction_type", 0),
            transaction.get("merchant_category", 0),
            transaction.get("time_of_day", 0),
            transaction.get("feature_5", 0),
            transaction.get("feature_6", 0),
            transaction.get("feature_7", 0),
            transaction.get("feature_8", 0),
            transaction.get("feature_9", 0),
            transaction.get("feature_10", 0),
            transaction.get("feature_11", 0),
            transaction.get("feature_12", 0),
            transaction.get("feature_13", 0),
            transaction.get("feature_14", 0),
            transaction.get("feature_15", 0)
        ]])

        # Ensure input matches model expectations
        if transaction_vector.shape[1] != fraud_model.n_features_in_:
            return {"error": f"Expected {fraud_model.n_features_in_} features, got {transaction_vector.shape[1]}"}

        prediction = fraud_model.predict(transaction_vector)
        return {"fraud_risk": "high" if prediction[0] == -1 else "low"}

    except Exception as e:
        logging.error(f"❌ Error in fraud detection: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}


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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
