from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm_groq = ChatGroq(model_name="llama3-70b-8192")

class Chatbot:
    prompt = """You are a helpful life assistant. Your name is Mitra! Be polite and friendly. Use the memory provided to provide a
            factually correct clear information. Help them understand meaning of life. Be very concise and under 3 sentences."""

    def __init__(self):
      self.messages = [] # setting up a basic memory
      self.messages.append({"role": "system", "content": self.prompt})

    def __call__(self, user_message):
      self.messages.append({"role": "user", "content": user_message})
      ai_message = llm_groq.invoke(self.messages)
      self.messages.append({"role": "assistant", "content": ai_message.content})
      return ai_message.content + "\n" + "Tokens used: " + str(ai_message.response_metadata["token_usage"]["total_tokens"])