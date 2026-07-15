import requests

API_URL = "https://customer-churn-prediction-dashboard-6qfw.onrender.com/predict"


def predict_churn(data):
    try:
        response = requests.post(API_URL, json=data, timeout=30)

        if response.status_code != 200:
            print(response.text)
            return {
                "error": response.text
            }

        return response.json()

    except Exception as e:
        return {
            "error": str(e)
        }