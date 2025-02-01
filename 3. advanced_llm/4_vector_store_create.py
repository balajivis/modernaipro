import os
os.environ['USER_AGENT'] = 'modernai_pro'

from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from PyPDF2 import PdfReader

import gradio as gr



# 2. Parse PDF
reader = PdfReader('arso.pdf')
text = ""
for i in range(0, len(reader.pages)):
    page = reader.pages[i]
    text += page.extract_text() + " "

# 3. Split document
documents = [Document(page_content=text, metadata={"source": "local"})]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40)
all_splits = text_splitter.split_documents(documents)

# 4. Setup the embeddings database
embedding_model = "mxbai-embed-large"
embeddings = OllamaEmbeddings(model=embedding_model)
collection = Chroma.from_documents(
    documents=all_splits, embedding=embeddings, persist_directory="doc_vectors")

# 5. Add more items to the collection
loader = WebBaseLoader(web_path=("https://mitrarobot.com/"))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40)
splits = text_splitter.split_documents(docs)

for s in splits:
    collection.add_documents([s])

# 6. Build a simple search


def language_chat(message, history):
    docs = collection.similarity_search(message, k=5)
    return docs[0].page_content + docs[1].page_content


demo = gr.ChatInterface(
    language_chat, title="Vector DB search", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch()
