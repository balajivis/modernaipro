import pandas as pd
import json

# Load the CSV into a DataFrame
df = pd.read_csv('../data/fine_tuning_recipe.csv')

# Initialize an empty list for the JSONL strings
jsonl_strings = []

# Iterate over each row of the DataFrame
for index, row in df.iterrows():
    # Formatting the conversation in the required format:
    # System message
    system_section = f"system\n\n{{{{ {row['system']} }}}}"
    # User message
    user_section = f"user\n\n{{{{ {row['user']} }}}}"
    # Assistant message
    assistant_section = f"assistant\n\n{{{{ {row['assistant']} }}}}"

    # Concatenating the formatted sections
    conversation_formatted = f"{system_section}\n{user_section}\n{assistant_section}"

    # Wrap the formatted conversation with {"text": ...}
    wrapped_data = {"text": conversation_formatted}

    # Serialize the wrapped data to a JSON formatted string
    jsonl_string = json.dumps(wrapped_data, ensure_ascii=False)

    # Append the final JSONL string to our list
    jsonl_strings.append(jsonl_string)

# Save the JSONL strings to a file
with open('data/recipe_ft.jsonl', 'w', encoding='utf-8') as file:
    for jsonl_string in jsonl_strings:
        file.write(jsonl_string + "\n")
