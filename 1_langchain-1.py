from langchain_community.llms import Ollama
llm = Ollama(model="phi3")

for chunks in llm.stream("Tell me the most interesting fact about India."):
    print(chunks, end='')
