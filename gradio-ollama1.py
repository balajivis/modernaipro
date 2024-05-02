import gradio as gr
import requests
import json


def chat_stream(message):
    """Streams responses from an API incrementally."""
    full = ""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen',
                'prompt': message,
                'context': []
            },
            stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                body = json.loads(line.decode('utf-8'))
                response_part = body.get('response', '')
                full += response_part
                if response_part:
                    yield [(message, full)]
                if 'error' in body:
                    yield [(message, f"Error: {body['error']}")]
                    break
                if body.get('done', False):
                    break
    except Exception as e:
        yield [(message, f"Error: {str(e)}")]


def chat_interface(message, history):
    """Generator to handle chat interface and manage chat history."""
    for items in chat_stream(message):
        for input, output in items:
            last_input = input
            last_output = output
        yield output


demo = gr.ChatInterface(
    chat_interface, title="Modern AI Pro", fill_height=True, theme='Taithrah/Minimal').queue()

if __name__ == "__main__":
    demo.launch(share=True)
