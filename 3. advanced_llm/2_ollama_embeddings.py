import ollama
text = "Idly is a south indian dish."
response = ollama.embeddings(model="llama3.2", prompt=text)
print(response)
