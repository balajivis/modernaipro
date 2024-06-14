# This time we are demonstrating:
# 1. LangFuse monitoring by logging responses
# 2. Switching different LLMs (Ollama and Groq)
# 3. Doing theming of Chatbot

import random
import gradio as gr
from langfuse.callback import CallbackHandler
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable
langfuse_handler = CallbackHandler()  # This will enable monitoring


def language_chat(message, history):
    llm = Ollama(model="gemma:2b")
    llm2 = ChatGroq(model_name="llama3-70b-8192")

    if "cricket" in message or "recipe" in message:
        choice = "Ollama"
    else:
        choice = "Groq"
    # choice = random.choice(["Ollama", "Groq"])
    print(f"Chosen model is {choice}")
    if choice == "Ollama":
        response = llm.invoke(
            message, config={"callbacks": [langfuse_handler]})
    else:
        response = llm2.invoke(
            message, config={"callbacks": [langfuse_handler]}).content
    return response


# Let's also theme it this time. Not the boring old interface.
demo = gr.ChatInterface(
    language_chat, title="LLM Switcher Modern AI Pro", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch()
