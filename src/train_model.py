import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib
import numpy as np

# Load final dataset
df = pd.read_csv("data/final_data.csv")

# Separate target
X = df.drop("Price", axis=1)
y = df["Price"]

# Encode categorical columns
categorical_cols = ["Location", "Condition", "Type"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

# Save model and encoders
joblib.dump(model, "models/price_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Model and encoders saved successfully.")
