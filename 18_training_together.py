from together import Together
import os
from dotenv import load_dotenv
load_dotenv()

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

resp = client.files.upload(file="data/recipe_ft.jsonl")

file_id = resp.model_dump()['id']

resp = client.fine_tuning.create(
    training_file=file_id,
    model='meta-llama/Meta-Llama-3-8B',
    n_epochs=3,
    n_checkpoints=1,
    batch_size=4,
    learning_rate=1e-5,
)

print(resp)
