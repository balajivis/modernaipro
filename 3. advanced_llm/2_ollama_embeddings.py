import ollama
text = "Idly is a south indian dish."
response = ollama.embeddings(model="gemma:2b", prompt=text)
print(response)
