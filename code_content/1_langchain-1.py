# Getting the host running Ollama
import os
ollama_host = os.environ["OLLAMA_HOST"] or "localhost"
base_url = f"http://{ollama_host}:11434"

from langchain_community.llms import Ollama

llm = Ollama(model="phi3", base_url=base_url)

for chunks in llm.stream("Tell me the most interesting fact about India."):
    print(chunks, end='')
