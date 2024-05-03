import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# Model names
models = {
    "mixtral-8x7b-32768": "Mixtral 8x7b 32768",
    "llama3-70b-8192": "Llama3 70b 8192",
    "gemma-7b-it": "Gemma 7b IT"
}

# Fixed chat setup
system = "You are a helpful assistant."
human = "{text}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system), ("human", human)])

# Test each model
for model_name, model_display_name in models.items():
    chat = ChatGroq(temperature=0, model_name=model_name)
    chain = prompt_template | chat

    start_time = time.time()
    response = chain.invoke({"text": "Write a haiku in 2 sentences."})
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Output the results
    print(f"Model: {model_display_name}")
    print(f"Response: {response}")
    print(f"API call latency: {elapsed_time:.3f} seconds\n")
