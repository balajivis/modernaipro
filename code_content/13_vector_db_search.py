# pip install chromadb
# ollama pull mxbai-embed-large

import ollama
import chromadb
import gradio as gr

documents = [
    "Biryani is a fragrant rice dish cooked with marinated meat, typically chicken or mutton, and seasoned with a variety of spices such as saffron and turmeric. It originated among the Muslim communities of the Indian subcontinent.",
    "Masala Dosa is a South Indian specialty consisting of a crispy, rice batter crepe filled with a spicy mix of mashed potatoes. It's served with a trio of chutneys and sambar, a lentil-based stew.",
    "Rogan Josh is a type of curry hailing from Kashmir. It features braised lamb chunks cooked with a gravy based on browned onions, yogurt, garlic, ginger, and aromatic spices like cloves, bay leaves, cardamom, and cinnamon.",
    "Pani Puri, also known as Golgappa, is a popular street snack that consists of a round, hollow puri filled with a mixture of flavored water (pani), tamarind chutney, chili, chaat masala, potato, onion, and chickpeas.",
    "Palak Paneer is a North Indian curry made from spinach and paneer (Indian cottage cheese) in a thick curry sauce based on pureed spinach, seasoned with garlic, garam masala, and other spices.",
    "Aloo Gobi is a vegetarian dish from the Indian subcontinent made with potatoes (aloo), cauliflower (gobi), and Indian spices. It is yellowish in color due to the use of turmeric, and sometimes contains kalonji and curry leaves.",
    "Samosas are a popular teatime snack and are often served as an appetizer. They are deep-fried pastries with a savory filling, such as spiced potatoes, onions, peas, and lentils."
]

client = chromadb.PersistentClient(path="vectors.db")
collection = client.create_collection(name="docs")
embedding_model = "mxbai-embed-large"
# store each document in a vector embedding database
for i, d in enumerate(documents):
    response = ollama.embeddings(model=embedding_model, prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

# search items with the query


def language_chat(message, history):
    response = ollama.embeddings(prompt=message, model=embedding_model)
    results = collection.query(
        query_embeddings=[response["embedding"]], n_results=1)
    return results['documents'][0][0]


demo = gr.ChatInterface(
    language_chat, title="Vector DB search", theme='Taithrah/Minimal')

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
