# Getting the host running Ollama
import os
ollama_host = os.environ["OLLAMA_HOST"] or "localhost"
base_url = f"http://{ollama_host}:11434"

from PIL import Image
from io import BytesIO
import base64
from langchain_community.llms import Ollama

buffered = BytesIO()
pil_image = Image.open("/workspace/data/senior.jpeg")
pil_image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

llm = Ollama(model="llava", base_url=base_url).bind(images=[img_str])
print(llm.invoke("Identify if any emergency event has happened. If yes, tell me the\
                  details. \
                 If no, leave it. Don't hallucinate"))
