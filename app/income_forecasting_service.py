import os
import joblib
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable

# Register custom MSE function
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Define paths
base_dir = os.getcwd()
models_dir = os.path.join(base_dir, "models")
lstm_model_path = os.path.join(models_dir, "lstm_model.h5")
scaler_path = os.path.join(models_dir, "scaler.pkl")

# Load Model & Scaler
if not os.path.exists(lstm_model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ ERROR: One or more LSTM model files are missing!")

# Ensure the model recognizes the custom loss function
lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects={'mse': mse})
scaler = joblib.load(scaler_path)

def forecast_income_expense(history):
    """
    Predicts future income and expenses using LSTM.

    :param history: list of past transactions (salary, expenses)
    :return: dict with predicted salary and expenses
    """

    # Debug: Print raw history data
    print("🔍 Raw History Input:", history)

    # Convert to NumPy array and reshape
    history_array = np.array(history).reshape(-1, 2)

    # Debug: Print shape of history array before scaling
    print("📝 History Array Shape Before Scaling:", history_array.shape)

    # Scale input data using the same scaler from training
    scaled_input = scaler.transform(history_array)

    # Debug: Print scaled input values
    print("📊 Scaled Input Data:", scaled_input)

    # Reshape for LSTM input (batch_size=1, timesteps=len(history), features=2)
    input_data = scaled_input.reshape(1, len(history), 2)

    # Debug: Print final input shape
    print("🚀 Input Shape to Model:", input_data.shape)

    # Predict
    prediction = lstm_model.predict(input_data)[0]

    # Debug: Print raw model predictions
    print("🤖 Raw Model Prediction:", prediction)

    # Check if predictions contain unexpected values
    if np.any(np.isnan(prediction)) or np.any(prediction < -1000):
        raise ValueError("⚠️ ERROR: Model produced NaN or highly negative values!")

    # Inverse transform to original scale
    predicted_scaled = np.array([prediction])  # Wrap in list to match scaler input format
    predicted_values = scaler.inverse_transform(predicted_scaled)

    # Debug: Print final inverse-transformed values
    print("✅ Inverse Transformed Prediction:", predicted_values)

    return {
        "predicted_salary": float(predicted_values[0][0]),
        "predicted_expenses": float(predicted_values[0][1])
    }
