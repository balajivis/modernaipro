import gradio as gr
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

prompt = """
You are an assistant for question-answering tasks for Mitra Robot customer support. 
Use the following pieces of retrieved context to answer the question. 
Use three sentences maximum and keep the answer concise.
"""

# 1. Load our DB
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db = Chroma(persist_directory="./doc_vectors", embedding_function=embeddings)
llm = Ollama(model="qwen")

# 2. set up our chat


def language_chat(message, history):
    docs = db.similarity_search(message)
    retrieved_string = "\n\n".join(doc.page_content for doc in docs)
    # print("I got these results for the message: ", message, retrieved_string)

    query = prompt + "The context is: " + \
        retrieved_string + "The question is :" + \
        message

    print(query)
    return llm.invoke(query)


demo = gr.ChatInterface(
    language_chat, title="Mitra Robot RAG", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch()
