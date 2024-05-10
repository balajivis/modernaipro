from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = WebBaseLoader(web_path=("https://mitrarobot.com/"))
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

for s in splits:
    print(s.page_content)
