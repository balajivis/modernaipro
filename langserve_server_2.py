from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Deque, List, Union

from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable

# 1. Setup the FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

# 3. Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Declare a chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful, professional assistant named Cob."),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | ChatAnthropic(
    temperature=0, model_name="claude-3-sonnet-20240229")


class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


add_routes(
    app,
    chain.with_types(input_type=InputChat),
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)
