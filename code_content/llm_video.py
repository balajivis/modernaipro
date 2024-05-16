# Getting the host running Ollama
import os
ollama_host = os.environ["OLLAMA_HOST"] or "localhost"
base_url = f"http://{ollama_host}:11434"

# This time we are demonstrating:
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
import gradio as gr

llm = Ollama(model="qwen", base_url=base_url)


def language_chat(message, history):
    response = llm.invoke(message)
    return response


demo = gr.ChatInterface(
    language_chat, title="LLM Evaluator Modern AI Pro", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
