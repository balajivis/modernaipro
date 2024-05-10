from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable

llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
print(llm.invoke("Write 2 sentences about India"))
