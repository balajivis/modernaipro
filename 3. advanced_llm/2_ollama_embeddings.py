import ollama
text = "Idly is a south indian dish."
response = ollama.embeddings(model="gemma2:2b", prompt=text)
print(response)
