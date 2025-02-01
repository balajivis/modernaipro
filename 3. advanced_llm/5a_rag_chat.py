import json
import requests
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import os
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Streamlit UI
st.title("👨‍🎓 Modern AI Pro: Full RAG System")

# Initialize embeddings
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
PERSIST_DIRECTORY = "./chroma_db"




# Check if vector store exists and load it
if not os.path.exists(PERSIST_DIRECTORY):
    st.error("No vector store found! Please run the indexing step first.")
    st.stop()

# Load vector store with status
with st.status("Loading vector store...", expanded=True) as status:
    vector = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name="document_chunks"  # Make sure this matches your indexing script
    )
    total_docs = vector._collection.count()
    status.update(label=f"Vector store loaded successfully with {total_docs} documents!", state="complete")

# Display vector store info
st.info(f"📚 Vector Store Statistics:\n"
        f"- Total documents: {total_docs}\n"
        f"- Location: {PERSIST_DIRECTORY}")

# 1. Create retriever with status update
with st.status("Configuring retrieval settings...", expanded=True) as status:
    retriever = vector.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 3}
    )
    status.update(label="Retriever configured to fetch top 3 most relevant chunks", state="complete")

# 2. Initialize LLM with model selection
with st.status("Initializing Large Language Model...", expanded=True) as status:
    # Get list of available Ollama models
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            status.update(label=f"Found {len(models)} available models")
        else:
            models = ["llama3.2:1b"]  # fallback default
            status.update(label="Couldn't fetch models, using default", state="error")
    except Exception as e:
        models = ["llama3.2:1b"]  # fallback default
        status.update(label=f"Error fetching models: {str(e)}", state="error")
    
    # Create dropdown for model selection
    selected_model = st.selectbox(
        "Select LLM Model",
        models,
        index=models.index("llama3.2:1b") if "llama3.2:1b" in models else 0,
        help="Choose from available Ollama models"
    )
    
    # Initialize the selected model
    llm = Ollama(model=selected_model)
    status.update(label=f"LLM ({selected_model}) initialized successfully", state="complete")

# 3. Enhanced prompt for the Modern AI Pro class
prompt = PromptTemplate.from_template("""You are an expert AI  assistant. Your task is to provide 
comprehensive answers based on the given context. Follow these guidelines:

1. Use the retrieved context to formulate your answer
2. Explain concepts clearly and provide examples when relevant
3. If the context doesn't contain enough information, acknowledge the limitations
4. Highlight key AI/ML concepts when they appear in the response
5. Maintain academic tone while being engaging

Context: {context}

Question: {input}

Educational Response:""")

# 4. Create chains with status updates
with st.status("Setting up RAG pipeline...", expanded=True) as status:
    combine_documents_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    qa_chain = create_retrieval_chain(retriever, combine_documents_chain)
    status.update(label="RAG pipeline ready for queries", state="complete")

# 5. User Interface
st.write("### Ask Questions About Your Document")
st.write("The system will retrieve relevant chunks and generate an educational response.")
user_input = st.text_input("Enter your question:")

if user_input:
    # Show retrieval and generation progress
    with st.status("Processing your question...", expanded=True) as status:
        # First update for retrieval
        status.update(label="Retrieving relevant chunks from document...")
        retrieved_docs = retriever.get_relevant_documents(user_input)
        status.update(label=f"Retrieved {len(retrieved_docs)} relevant chunks")
        
        # Second update for generation
        status.update(label="Generating the LLM response...")
        response = qa_chain.invoke({"input": user_input})
        
        # Final update
        status.update(label="Response generated successfully!", state="complete")

    # 10. Display response with formatting
    st.write("### Educational Response")
    st.write(response["answer"])
    
    # 11. show retrieved chunks for transparency
    with st.expander("View Retrieved Context Chunks"):
        for i, doc in enumerate(retrieved_docs, 1):
            st.write(f"**Chunk {i}:**")
            st.write(doc.page_content)
            st.write("---")