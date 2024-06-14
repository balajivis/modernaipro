import pandas as pd
import json

df = pd.read_csv('../data/Cleaned_Indian_Food_Dataset.csv')

assistant_responses = []

for index, row in df.iterrows():
    recipe_dict = {
        "greeting": "Namaste! Thanks for asking.",
        "recipeName": row["TranslatedRecipeName"],
        "imageUrl": row["image-url"],
        "ingredients": row["TranslatedIngredients"],
        "instructions": row["TranslatedInstructions"],
        "attributionUrl": row["URL"],
        "cuisineType": row["Cuisine"],
        "prepTimeMinutes": row["TotalTimeInMins"]
    }

    recipe_json = json.dumps(recipe_dict)

    assistant_responses.append(recipe_json)

# Create a new DataFrame for the final CSV
final_df = pd.DataFrame({
    "system": "You are a helpful cooking robot tasked with helping with Indian recipes.",
    "user": df["TranslatedRecipeName"],
    "assistant": assistant_responses
})

# Save the new DataFrame to a CSV file
final_df.to_csv('../data/fine_tuning_recipe.csv', index=False)
