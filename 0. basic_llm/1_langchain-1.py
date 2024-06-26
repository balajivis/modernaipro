# Note:
# Use something like this: conda create --name modernaipro python=3.11 --file requirements.txt
# conda activate modernaipro

from langchain_community.llms import Ollama
llm = Ollama(model="gemma:2b") # try qwen2 if you have that model

for chunks in llm.stream("Tell."):
    print(chunks, end='\n')
