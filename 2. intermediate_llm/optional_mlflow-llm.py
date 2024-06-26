import mlflow
import openai
import pandas as pd
from professionalism_metric import professionalism_metric

from dotenv import load_dotenv
load_dotenv()  # will search for .env file in local folder and load variable
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Evaluate LLMs")
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What does Mitra Robot do?",
            "What is a likability index used in Mitra Robot",
        ],
        "ground_truth": [
            """
                Mitra Robot is a Senior Care robot that helps older adults live independently at home through voice interactions and autonomous mobility. It provides companionship, medication reminders, video calls with family, and emergency response if needed.
            """,
            """
            Likability index measures how well a robot is liked by seniors and used various factors like empathy, social skills and ease of use. It is an important metric for robots designed to interact regularly with older adults.
            """
        ],
    }
)

with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4",
        task=openai.ChatCompletion,
        artifact_path="model",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        targets="ground_truth",
        extra_metrics=[mlflow.metrics.genai.answer_correctness(), mlflow.metrics.genai.answer_similarity(), professionalism_metric, mlflow.metrics.latency(
        ), mlflow.metrics.genai.answer_similarity()],
    )
