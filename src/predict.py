import requests

API_URL = "http://127.0.0.1:8000/predict"


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