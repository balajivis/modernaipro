from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes

# 1. Setup the FastAPI app
app = FastAPI(
    title="Modern AI Pro Server",
    version="1.0",
    description="For class coding",
)


# 2. Setup the LLM call
prompt = ChatPromptTemplate.from_template("Generate tweet on this topic: {topic}")
add_routes(
    app,
    prompt | Ollama(model="gemma2:2b"),
    path="/tweet-gen",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
