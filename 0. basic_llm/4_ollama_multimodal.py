from PIL import Image
from io import BytesIO
import base64
from langchain_community.llms import Ollama

buffered = BytesIO()
pil_image = Image.open("../data/senior.png")
pil_image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

llm = Ollama(model="llava").bind(images=[img_str])
print(llm.invoke("Tell me if there is an emergency event in this picture."))
