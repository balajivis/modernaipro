# 1. Do a pip install of llama index: pip install langchain-groq
# 2. In an .env file setup GROQ_API_KEY=
import time
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Model names
models = {
    "mixtral-8x7b-32768": "Mixtral 8x7b 32768",
    "llama3-70b-8192": "Llama3 70b 8192",
    "gemma-7b-it": "Gemma 7b IT"
}

# Test each model
for model_name, model_display_name in models.items():
    llm = ChatGroq(model_name=model_name)

    start_time = time.time()
    response = llm.invoke(
        "Talk about your most favorite EPL football club in 2 sentences")
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Output the results
    print(f"Model: {model_display_name}")
    print(f"Response: {response}")
    print(f"API call latency: {elapsed_time:.3f} seconds\n")
