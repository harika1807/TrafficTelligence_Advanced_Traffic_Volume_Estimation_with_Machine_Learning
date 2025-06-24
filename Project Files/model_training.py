# model_training.py

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
data = pd.read_csv("traffic_data.csv")

# Replace 'None' string with actual NaN
data.replace("None", np.nan, inplace=True)

# Drop rows where target variable is missing
data.dropna(subset=["Traffic_volume"], inplace=True)

# Handle missing numeric values
numeric_cols = ['Temp', 'Rain', 'Snow']
num_imputer = SimpleImputer(strategy='mean')
data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

# Convert 'Date' and 'Time' columns to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S', errors='coerce')

# Extract date/time features
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Hour'] = data['Time'].dt.hour

# Drop original columns
data.drop(['Date', 'Time', 'Weather'], axis=1, errors='ignore', inplace=True)

# Replace all non-null holiday names with 1, and null/None/empty with 0
data['Holiday'] = data['Holiday'].apply(lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['none', ''] else 1)
data['Holiday'] = data['Holiday'].astype(int)

# Select only 8 required input features
features = ['Holiday', 'Temp', 'Rain', 'Snow', 'Year', 'Month', 'Day', 'Hour']
X = data[features]
y = data['Traffic_volume']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost Regressor
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# Evaluate
rf_r2 = r2_score(y_test, rf_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

print("Random Forest R2 Score:", rf_r2)
print("Random Forest RMSE:", rf_rmse)
print("XGBoost R2 Score:", xgb_r2)
print("XGBoost RMSE:", xgb_rmse)

# Choose the better model
best_model = rf_model if rf_r2 > xgb_r2 else xgb_model

# Save the model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")
