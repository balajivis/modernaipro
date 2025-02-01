# Apache license
import gradio as gr
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()

# Load the Document
prompt = """
Your name is Mitra. You are an assistant for question-answering tasks for Mitra Robot customer support. 
You need to use the following pieces of retrieved context to answer the question. 
Use 3-5 sentences maximum and keep the answer concise. Answer in a professional tone. 
Answer only robot related questions and nothing else. If they ask something like biriyani bring them back to the conversation
If there is a contact detail or anything you are offering make sure it exits.
Sales enquiries point it to sales@mitrarobot.com. 
"""

# Prompt + My policy document

# 1. Load our DB
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory="./doc_vectors", embedding_function=embeddings)
# llm = Ollama(model="llamad3.2")
llm = ChatGroq(model_name="llama-3.2-90b-text-preview", temperature=0.5)

# 2. set up our chat


def language_chat(message, history):
    docs = db.similarity_search(message)
    retrieved_string = "\n\n".join(doc.page_content for doc in docs)
    print("I got these results for the message: ", message, retrieved_string)

    query = prompt + ". The context is: " + \
        retrieved_string + "The question is :" + \
        message

    # print(history)
    print(query)
    return llm.invoke(query).content # remove .content if using ollama


demo = gr.ChatInterface(
    language_chat, title="Mitra Robot RAG", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch()
