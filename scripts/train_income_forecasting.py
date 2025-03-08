import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# Define paths
base_dir = os.getcwd()  # Get current working directory
data_path = os.path.join(base_dir, 'data', 'customer_income_expense_history.csv')
model_dir = os.path.join(base_dir, 'models')

# Ensure models directory exists
os.makedirs(model_dir, exist_ok=True)

# Load Data
df = pd.read_csv(data_path)

# Validate columns exist
required_columns = ['salary_deposits', 'total_expenses']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"❌ ERROR: Missing columns in dataset: {missing_columns}")

# Normalize Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['salary_deposits', 'total_expenses']])

# Create Sequences (Ensure at least 7 data points exist)
X, y = [], []
if len(scaled_data) > 6:
    for i in range(len(scaled_data) - 6):
        X.append(scaled_data[i:i+6])   # Last 6 months of data
        y.append(scaled_data[i+6])     # Predict next month

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(6, 2)),
    LSTM(50, activation='relu'),
    Dense(2)
])

# Compile & Train Model
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16, verbose=1)

# Save Model & Scaler
model_path = os.path.join(model_dir, 'lstm_model.h5')
scaler_path = os.path.join(model_dir, 'scaler.pkl')

model.save(model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ Income Forecasting Model Trained and Saved at {model_path}")
