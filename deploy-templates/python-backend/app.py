from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
from chatbot import Chatbot

class InvokeRequest(BaseModel):
    input: str
    config: Dict[str, Any]
    kwargs: Dict[str, Any]
    

bot = Chatbot()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def hello_world():
    return "Hello,World"


@app.post("/life_chat")
async def invoke(request: InvokeRequest):
    try:
        response = bot(request.input)
        print(response)
        return {"output": response}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)