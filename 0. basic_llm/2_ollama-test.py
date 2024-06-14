import ollama

prompt = "What is your favorite marketing book? Pick one and write 2 sentences about it"

role = "user"

messages = [{
    'role': role,
    'content': prompt,
}]

stream = ollama.chat(model='qwen2', messages=messages, stream=True)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
