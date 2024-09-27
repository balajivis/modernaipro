import ollama

prompt = "Give me marketing into 101"

role = "user"

messages = [{
    'role': role,
    'content': prompt,
}]

stream = ollama.chat(model='llama3.2', messages=messages, stream=True)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
