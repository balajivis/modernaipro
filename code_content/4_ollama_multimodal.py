from PIL import Image
from io import BytesIO
import base64
from langchain_community.llms import Ollama

buffered = BytesIO()
pil_image = Image.open("senior.jpeg")
pil_image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

llm = Ollama(model="llava").bind(images=[img_str])
print(llm.invoke("Identify if any emergency event has happened. If yes, tell me the\
                  details. \
                 If no, leave it. Don't hallucinate"))
