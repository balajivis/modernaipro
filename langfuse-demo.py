from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langfuse.callback import CallbackHandler
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()  # will search for .env file in local folder and load variable

llm = Ollama(model="llama3")
llm2 = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | llm
chain2 = prompt | llm2
#######

langfuse_handler = CallbackHandler(
    public_key="pk-lf-57e08851-27c7-475b-a533-b5e367c0fa40",
    secret_key="sk-lf-53abc109-6f09-4711-8585-c6a78037a5e0",
    host="https://cloud.langfuse.com"
)

print(chain.invoke({"topic": "bears"}, config={
      "callbacks": [langfuse_handler]}))
print(chain2.invoke({"topic": "bears"}, config={
      "callbacks": [langfuse_handler]}))
