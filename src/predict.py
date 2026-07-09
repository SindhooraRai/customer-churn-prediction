import requests

API_URL = "http://127.0.0.1:8000/predict"


def predict_churn(data):
    """
    Send customer data to the FastAPI server
    and return the prediction.
    """

    response = requests.post(API_URL, json=data)

    return response.json()