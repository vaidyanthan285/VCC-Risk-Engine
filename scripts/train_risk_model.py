import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define base directory dynamically
base_dir = os.getcwd()  # Get current working directory
data_path = os.path.join(base_dir, 'data', 'customer_transaction_data.csv')
model_dir = os.path.join(base_dir, 'models')

# Ensure models directory exists
os.makedirs(model_dir, exist_ok=True)

# Load Data
df = pd.read_csv(data_path)

# Validate 'risk_level' column exists
if 'risk_level' not in df.columns:
    raise ValueError("❌ ERROR: 'risk_level' column not found in dataset!")

# Convert categorical target variable to numerical
df['risk_level'] = df['risk_level'].map({'low': 0, 'medium': 1, 'high': 2})

# Feature Engineering
df['savings_ratio'] = (df['salary_deposits'] - df['total_expenses']) / (df['salary_deposits'] + 1e-9)  # Avoid division by zero
df['spending_variability'] = np.abs(df['total_expenses'].pct_change()).fillna(0)

# Define features & labels
X = df.drop(columns=['risk_level'])
y = df['risk_level']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=2, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Train Model with Best Parameters
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)

# Save Model
model_path = os.path.join(model_dir, 'risk_model.pkl')
joblib.dump(best_model, model_path)

print(f"✅ Risk Model Trained and Saved at {model_path}")
