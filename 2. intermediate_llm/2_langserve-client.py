import requests

response = requests.post(
    "http://localhost:8000/tweet-gemma/invoke",
    json={'input': {'topic': "the beauty of ocean"}})

print(response.json()['output'])
