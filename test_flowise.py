import requests

API_URL = "http://localhost:3000/api/v1/prediction/308d0b02-74eb-46d8-830c-ab9685ccc217"


def query(payload):
    response = requests.post(API_URL, json=payload)
    return response.json()


output = query({
    "question": "Hey, how are you?",
})
