"""
TASK 2: Predictive Linear Regression Model Pipeline
Dataset: California Housing (built into scikit-learn, no download needed)
"""

import numpy as np
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target  # median house value (in $100,000s)

print("Dataset shape:", X.shape)
print(X.head())

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocessing + Model pipeline
# (mean imputation handles any missing values; scaling helps Ridge regression converge properly)
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# 4. Train
pipeline.fit(X_train, y_train)

# 5. Evaluate
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR2 Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# 6. Save the trained pipeline (so the app can load it without retraining)
joblib.dump(pipeline, "house_price_model.joblib")
print("\nModel saved as house_price_model.joblib")
