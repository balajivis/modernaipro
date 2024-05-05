import ollama
import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()

together_key = os.environ.get("TOGETHER_API_KEY")

url = "https://api.together.xyz/v1/completions"

query = """
Andhra Style chutney. 
"""

payload = {
    "model": "balaji@mitrarobot.com/Meta-Llama-3-8B-2024-04-20-17-04-24",
    "prompt": query,
    "max_tokens": 1200,
    "top_p": 0.7,
    "top_k": 50,
    "temperature": 0.71,
    "repetition_penalty": 1.28
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": f"Bearer {together_key}"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)
