# Getting the host running Ollama
import os
ollama_host = os.environ["OLLAMA_HOST"] or "localhost"
base_url = f"http://{ollama_host}:11434"

import gradio as gr
from PIL import Image
from io import BytesIO
import base64
from langchain_community.llms import Ollama


def process_file(message, history):
    if message["files"] and message["text"]:
        # Open the first file (assuming it's an image)
        image_file = message["files"][0]
        pil_image = Image.open(image_file)

        # Convert RGBA to RGB if necessary
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')

        # Save the image to a buffer
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        llm = Ollama(model="llava", base_url=base_url).bind(images=[img_str])
        response = llm.invoke(message["text"])
        return f"Model response: {response}"


demo = gr.ChatInterface(
    fn=process_file, title="Mitra Robot Emergency Checker", multimodal=True)

demo.launch(share=False, server_name="0.0.0.0")
