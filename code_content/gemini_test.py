import os
import time
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")

start_time = time.time()
response = model.generate_content("What is the meaning of life?")
end_time = time.time()
elapsed_time = end_time - start_time

print(response.text)
print(f"API call latency: {elapsed_time:.3f} seconds\n")
