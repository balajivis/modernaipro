import gradio as gr
import pandas as pd
import pickle
import numpy as np

# Load the model from disk
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)


def predict_price(OverallQual, GrLivArea, GarageCars, TotalSquareFeet, TotalBath, SqFtPerRoom, YearBuilt):
    # Create a DataFrame with the necessary features
    data = {
        'OverallQual': [OverallQual],
        'TotalSquareFeet': [TotalSquareFeet],
        'TotalBath': [TotalBath],
        'GrLivArea': [GrLivArea],
        'SqFtPerRoom': [SqFtPerRoom],
        'GarageCars': [GarageCars],
        'YearBuilt': [YearBuilt]
    }
    input_df = pd.DataFrame(data)

    # Predict the price using the model
    predicted_price = model.predict(input_df)[0]
    return f"${predicted_price:,.2f}"


# Define the Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, value=5,
                  label="Overall Quality", show_label=True),
        gr.Slider(500, 4000, value=1500,
                  label="Above grade (ground) living area square feet"),
        gr.Slider(0, 4, value=2, label="Number of cars in the garage"),
        gr.Slider(1000, 4000, value=2500, label="Total square feet"),
        gr.Slider(1, 6,  value=3, label="Total number of bathrooms"),
        gr.Slider(500, 3000, value=1000, label="Square feet per room"),
        gr.Slider(1900, 2024, value=2010, label="Year built")
    ],
    outputs="text",
    title="Modern AI Pro: Home Price Prediction Model",
    description="Adjust the sliders to predict the home prices."
)

# Launch the app to the world
# https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
iface.launch(share=False)

