import ollama
text = "Pledge: India is my country and all Indians are my brothers and sisters."
response = ollama.embeddings(model="qwen", prompt=text)
print(response)
