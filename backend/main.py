from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np

from src.feature_engineering import engineer_features

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0"
)

# -----------------------------
# Load trained model
# -----------------------------
# -----------------------------
# Load trained model
# -----------------------------
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(
    os.path.join(BASE_DIR, "models", "random_forest_churn.pkl")
)

feature_names = joblib.load(
    os.path.join(BASE_DIR, "models", "feature_names.pkl")
)

# SHAP Explainer
explainer = shap.TreeExplainer(model)


# -----------------------------
# Customer Input Schema
# -----------------------------
class CustomerFeatures(BaseModel):
    customerID: str = "0001"

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict_churn(customer: CustomerFeatures):

    # Convert request to DataFrame
    df = pd.DataFrame([customer.model_dump()])

    # Dummy target so preprocessing works
    df["Churn"] = "No"

    # Apply same preprocessing as training
    df = engineer_features(df)

    # Remove target column
    df = df.drop("Churn", axis=1)

    # Add any missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Keep exact training column order
    df = df[feature_names]

    # Prediction
   
    prob = model.predict_proba(df)[0][1]
   

    # SHAP values
    try:
        explanation = explainer(df)

        # Get SHAP values for first prediction
        shap_vals = explanation.values[0]

        # If multiclass or extra dimension exists
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[:, 1]

        # Convert to Python floats
        top_reasons = sorted(
            zip(df.columns, shap_vals.tolist()),
            key=lambda x: abs(float(x[1])),
            reverse=True
        )[:5]

    except Exception as e:
        print("SHAP Error:", e)

        # Continue prediction even if SHAP fails
        top_reasons = []

    # Risk Level
    if prob >= 0.60:
        risk = "HIGH"
    elif prob >= 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "churn_probability": round(float(prob), 4),
        "risk_level": risk,
        "top_reasons": [
        {
            "feature": feature,
            "impact": round(float(impact), 4)
        }
        for feature, impact in top_reasons
    ]
    }
@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running!"
    }

    

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok"
    }