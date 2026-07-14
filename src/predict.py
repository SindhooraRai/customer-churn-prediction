import requests

API_URL = "https://customer-churn-prediction-dashboard-6qfw.onrender.com/predict"


def predict_churn(data):
    """
    Send customer data to the FastAPI server
    and return the prediction.
    """

    response = requests.post(API_URL, json=data)

    return response.json()