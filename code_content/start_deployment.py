import requests
import os

together_key = os.environ.get("TOGETHER_API_KEY")

url = "https://api.together.xyz/instances/start?model=balaji%40mitrarobot.com%2FMeta-Llama-3-8B-2024-04-20-17-04-24"

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {together_key}"
}

response = requests.post(url, headers=headers)

print(response.text)
