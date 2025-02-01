import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

PERSIST_DIRECTORY = "./chroma_db"

# Create the directory if it doesn't exist
if not os.path.exists(PERSIST_DIRECTORY):
    os.makedirs(PERSIST_DIRECTORY)

# Streamlit UI
st.title("👨‍🎓 Modern AI Pro: Add documents to vector store for RAG!!")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

# After the file upload and before processing
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # 1. Loading PDF
    with st.status("Loading PDF...", expanded=True) as status:
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()
        status.update(label=f"PDF loaded successfully! ({len(docs)} pages)", state="complete")

    # 2. Loading embedding model
    with st.status("Initializing embedding model...", expanded=True) as status:
        embedder = OllamaEmbeddings(model="mxbai-embed-large")
        status.update(label="Embedding model initialized!", state="complete")

    # 3. Splitting text
    with st.status("Splitting text into semantic chunks...", expanded=True) as status:
        text_splitter = SemanticChunker(embedder)
        documents = text_splitter.split_documents(docs)
        status.update(label=f"Text split into {len(documents)} semantic chunks", state="complete")

    # 4. Creating or loading vector store
    with st.status("Setting up vector store...", expanded=True) as status:
        if os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
            # Load existing database
            status.update(label="Loading existing vector store...")
            vector = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedder,
                collection_name="document_chunks"  
            )
            
            # Add new documents to existing database
            total_chunks = len(documents)
            if total_chunks > 0:
                progress_bar = st.progress(0)
                status.update(label="Adding new documents to existing vector store...")
                
                # Process documents in batches
                batch_size = 5
                for i in range(0, total_chunks, batch_size):
                    batch = documents[i:i + batch_size]
                    vector.add_documents(batch)
                    
                    # Update progress
                    progress = min((i + batch_size) / total_chunks, 1.0)
                    progress_bar.progress(progress)
                    status.update(label=f"Adding chunks: {min(i + batch_size, total_chunks)}/{total_chunks}")
                
                vector.persist()  # Make sure to persist after adding documents
                progress_bar.empty()
                status.update(label=f"Added {total_chunks} new chunks to existing vector store!", state="complete")
            else:
                status.update(label="Existing vector store loaded successfully!", state="complete")
        else:
            # Create new vector store
            status.update(label="Creating new vector store...")
            total_chunks = len(documents)
            progress_bar = st.progress(0)
            
            # Process documents in batches
            batch_size = 5
            for i in range(0, total_chunks, batch_size):
                batch = documents[i:i + batch_size]
                if i == 0:
                    vector = Chroma.from_documents(
                        documents=batch,
                        embedding=embedder,
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name="document_chunks"  # Added collection name for clarity
                    )
                else:
                    vector.add_documents(batch)
                
                # Update progress
                progress = min((i + batch_size) / total_chunks, 1.0)
                progress_bar.progress(progress)
                status.update(label=f"Embedding chunks: {min(i + batch_size, total_chunks)}/{total_chunks}")
            
            # Persist the database
            vector.persist()
            progress_bar.empty()
            status.update(label=f"Vector store created with {total_chunks} chunks and persisted!", state="complete")

        # Display total documents in database
        total_docs = vector._collection.count()
        st.write(f"📚 Total documents in vector store: {total_docs}")