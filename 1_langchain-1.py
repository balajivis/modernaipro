from langchain_community.llms import Ollama
llm = Ollama(model="qwen")

for chunks in llm.stream("Tell me the most interesting fact about India."):
    print(chunks, end='\n')
