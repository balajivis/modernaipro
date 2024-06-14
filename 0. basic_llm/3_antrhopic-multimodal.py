from pathlib import Path
import base64
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable

llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
print(llm.invoke("Write 2 sentences about India").content)

print("\n\nNow. We will look at multimodality.\n")
img_path = Path("../data/senior.png")
img_base64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
messages = [
    HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {
                    # langchain logo
                    "url": f"data:image/png;base64,{img_base64}",  # noqa: E501
                },
            },
            {"type": "text", "text": "Is there an emergency in this image?"},
        ]
    )
]

print(llm.invoke(messages).content)
