import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import pandas as pd
from src.feature_engineering import engineer_features


def test_feature_engineering_output_shape():

    df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    result = engineer_features(df)

    assert result.shape[0] == df.shape[0]
    assert "customerID" not in result.columns
    assert "Churn" in result.columns


def test_no_missing_values():

    df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    result = engineer_features(df)

    assert result.isnull().sum().sum() == 0


def test_churn_is_binary():

    df = pd.read_csv(
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    result = engineer_features(df)

    assert set(result["Churn"].unique()) == {0, 1}