import pandas as pd
import numpy as np


def engineer_features(df):
    """
    Perform feature engineering and preprocessing
    on the customer churn dataset.
    """

    # Create a copy
    df = df.copy()

    # -------------------------------------------------
    # Fix TotalCharges
    # -------------------------------------------------
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    # -------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------

    # Average monthly spending
    df["AvgMonthlySpend"] = (
        df["TotalCharges"] / (df["tenure"] + 1)
    )

    # Customer tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["new", "mid", "mature", "loyal"],
        include_lowest=True
    )

    # Number of premium services
    df["HasMultiServices"] = (
        (df["OnlineSecurity"] == "Yes").astype(int)
        + (df["TechSupport"] == "Yes").astype(int)
        + (df["StreamingTV"] == "Yes").astype(int)
    )

    # -------------------------------------------------
    # Binary Encoding
    # -------------------------------------------------

    binary_cols = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    for col in binary_cols:

        if col == "gender":
            df[col] = (df[col] == "Male").astype(int)

        else:
            df[col] = (df[col] == "Yes").astype(int)

    # -------------------------------------------------
    # One-Hot Encoding
    # -------------------------------------------------

    df = pd.get_dummies(
        df,
        columns=[
            "MultipleLines",
            "InternetService",
            "Contract",
            "PaymentMethod",
            "TenureGroup"
        ],
        drop_first=True
    )

    # -------------------------------------------------
    # Drop Unnecessary Columns
    # -------------------------------------------------

    if "customerID" in df.columns:
        df.drop(
            "customerID",
            axis=1,
            inplace=True
        )

    # -------------------------------------------------
    # Encode Target
    # -------------------------------------------------

    if "Churn" in df.columns:
        df["Churn"] = (
            df["Churn"] == "Yes"
        ).astype(int)

    return df