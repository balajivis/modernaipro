# This time we are demonstrating:
# 1. Langchain evaluation
# 2. LangFuse scoring
from langfuse.decorators import langfuse_context, observe
from langchain.evaluation import load_evaluator, Criteria
from langfuse import Langfuse

import gradio as gr
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable

eval_criteria = "conciseness"
infer_llm = Ollama(model="gemma2:2b")
eval_llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
langfuse = Langfuse()


@observe()
def language_chat(message, history):
    langfuse_handler = langfuse_context.get_current_langchain_handler()
    response = infer_llm.invoke(
        message, config={"callbacks": [langfuse_handler]})

    evaluator = load_evaluator(
        "criteria", llm=eval_llm, criteria=eval_criteria)

    eval_result = evaluator.evaluate_strings(
        prediction=response,
        input=message,
    )
    langfuse.score(name=eval_criteria, trace_id=langfuse_context.get_current_trace_id(),
                   value=eval_result["score"], comment=eval_result['reasoning'])

    return response


demo = gr.ChatInterface(
    language_chat, title="LLM Evaluator Modern AI Pro", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch()



