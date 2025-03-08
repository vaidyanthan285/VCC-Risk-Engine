import os
import joblib
import numpy as np
import tensorflow as tf

# Define paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")
lstm_model_path = os.path.join(models_dir, "lstm_model.h5")
scaler_path = os.path.join(models_dir, "scaler.pkl")

# Load Model & Scaler
if not os.path.exists(lstm_model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ ERROR: One or more LSTM model files are missing!")

lstm_model = tf.keras.models.load_model(lstm_model_path)
scaler = joblib.load(scaler_path)

def forecast_income_expense(history):
    """
    Predicts future income and expenses using LSTM.

    :param history: list of past transactions (salary, expenses)
    :return: dict with predicted salary and expenses
    """
    # Convert to NumPy array and reshape
    # input_data = np.array(history).reshape(1, len(history), 2)
    input_data = np.array(data["history"]).reshape(1, len(data["history"]), 2)


    # Predict
    prediction = lstm_model.predict(input_data)[0]

    # Inverse transform
    predicted_values = scaler.inverse_transform([prediction])

    return {
        "predicted_salary": predicted_values[0][0],
        "predicted_expenses": predicted_values[0][1]
    }
