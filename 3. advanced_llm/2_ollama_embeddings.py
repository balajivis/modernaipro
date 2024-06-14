import ollama
text = "Pledge: India is my country and all Indians are my brothers and sisters."
response = ollama.embeddings(model="gemma:2b", prompt=text)
print(response)
