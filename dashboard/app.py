import streamlit as st
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_churn

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer details below and click Predict.")
st.sidebar.header("Customer Details")

# -----------------------------
# Customer Inputs
# -----------------------------

customerID = st.sidebar.text_input("Customer ID", "0001")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    value=12
)

PhoneService = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

InternetService = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No"]
)

OnlineBackup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

DeviceProtection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

TechSupport = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

StreamingTV = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

StreamingMovies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

Contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

PaperlessBilling = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

TotalCharges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=900.0
)

# -----------------------------
# Predict Button
# -----------------------------

if st.sidebar.button("Predict Churn"):

    customer = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    result = predict_churn(customer)
    st.write(result)

    raw_probability = float(result["churn_probability"])
    st.write("Raw:", raw_probability)
    st.write("Progress:", raw_probability / 100)
    
    
    st.success("Prediction completed successfully! ✅")
    st.subheader("📋 Customer Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Customer ID:** {customerID}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Tenure:** {tenure} Months")
        st.write(f"**Contract:** {Contract}")

    with summary_col2:
        st.write(f"**Internet:** {InternetService}")
        st.write(f"**Monthly Charges:** ₹{MonthlyCharges}")
        st.write(f"**Total Charges:** ₹{TotalCharges}")
        st.write(f"**Payment Method:** {PaymentMethod}")

    

    st.subheader("📈 Prediction Results")
    if "churn_probability" not in result:
        st.error(result.get("error", "Prediction failed"))
        st.stop()

    raw_probability = float(result["churn_probability"])

    # Backend returns percentage (e.g. 45.435)
    probability = raw_probability / 100.0

    # Keep progress value between 0 and 1
    probability = max(0.0, min(probability, 1.0))

    risk = result["risk_level"]

    st.write("API Response:", result)
    st.write("Raw Probability:", raw_probability)
    st.write("Progress Value:", probability)

   
    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="📊 Churn Probability",
            value=f"{raw_probability:.2f}%"
        )

    with metric2:
        st.metric(
            label="🚨 Risk Level",
            value=risk
        )

    st.progress(probability)
   

    st.write(f"### Churn Probability: {probability*100:.2f}%")

    if risk == "HIGH":
        st.error(
            "🔴 HIGH RISK\n\n"
            "This customer has a high probability of churning. "
            "Retention actions should be taken immediately."
        )

    elif risk == "MEDIUM":
        st.warning(
            "🟡 MEDIUM RISK\n\n"
            "This customer may churn. Consider offering discounts "
            "or personalized support."
        )

    else:
        st.success(
            "🟢 LOW RISK\n\n"
            "This customer is likely to stay. Continue providing "
            "good service."
        )
    feature_map = {
        "InternetService_Fiber optic": "Fiber Optic Internet",
        "InternetService_No": "No Internet Service",
        "PaymentMethod_Electronic check": "Electronic Check Payment",
        "PaymentMethod_Mailed check": "Mailed Check Payment",
        "PaymentMethod_Credit card (automatic)": "Credit Card (Auto)",
        "Contract_One year": "One-Year Contract",
        "Contract_Two year": "Two-Year Contract",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Technical Support",
        "Partner": "Has Partner",
        "Dependents": "Has Dependents",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "AvgMonthlySpend": "Average Monthly Spend",
        "tenure": "Customer Tenure",
        "HasMultiServices": "Multiple Services"
    }

    st.subheader("📊 Top Factors Affecting Prediction")

    features = [
        reason["feature"]
        for reason in result["top_reasons"]
    ]

    impacts = [
        abs(reason["impact"])
        for reason in result["top_reasons"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.barh(features, impacts)

    ax.set_xlabel("SHAP Impact")
    ax.set_ylabel("Features")
    ax.set_title("Top Features Influencing Prediction")

    plt.tight_layout()

    st.pyplot(fig)