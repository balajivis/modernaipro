import requests

response = requests.post(
    "http://localhost:8000/tweet-phi3/invoke",
    json={'input': {'topic': "the beauty of ocean"}})

print(response.json()['output'])
