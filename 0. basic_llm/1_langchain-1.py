# Note:
# Use something like this: conda create --name modernaipro python=3.11 --file requirements.txt
# conda activate modernaipro

from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2") # try qwen2 / llama3 if you have that model


for chunks in llm.stream("History of india in 3 sentences"):
    print(chunks, end='\n')
