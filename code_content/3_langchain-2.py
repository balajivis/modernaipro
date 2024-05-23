from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
print(llm.invoke("Write 2 sentences about India"))
