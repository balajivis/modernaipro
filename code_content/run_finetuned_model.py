# download the model
#  zstd -d .zst model file
# tar -xvf tar_file -C cooking_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("cooking_model")

model = AutoModelForCausalLM.from_pretrained(
    "cooking_model",
    trust_remote_code=True,
).to(device)


def get_completion(input):
    input_ids = tokenizer.encode(input, return_tensors="pt")
    output = model.generate(input_ids.to(
        device), max_length=512, temperature=0.1)
    return tokenizer.decode(output[0], skip_special_tokens=True)


get_completion("Draft a hypothetical legal advice for a client based on the outcome of the provided case. entral Inland Water Transport Corporation Ltd. vs Brojo Nath Ganguly & Anr., 1986 AIR 1571, 1986 SCR (2) 278")
