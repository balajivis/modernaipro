import chromadb
import ollama
import gradio as gr
import os

# Path to the vector database
db_path = "vectors.db"
embedding_model = "mxbai-embed-large"
client = chromadb.PersistentClient(path=db_path)

if os.path.exists(db_path):
    collection = client.get_collection(name="docs")
else:
    collection = client.create_collection(name="docs")
    

    # List of documents
    documents = [
        "Biryani is a fragrant rice dish cooked with marinated meat, typically chicken or mutton, and seasoned with a variety of spices such as saffron and turmeric. It originated among the Muslim communities of the Indian subcontinent.",
        "Masala Dosa is a South Indian specialty consisting of a crispy, rice batter crepe filled with a spicy mix of mashed potatoes. It's served with a trio of chutneys and sambar, a lentil-based stew.",
        "Rogan Josh is a type of curry hailing from Kashmir. It features braised lamb chunks cooked with a gravy based on browned onions, yogurt, garlic, ginger, and aromatic spices like cloves, bay leaves, cardamom, and cinnamon.",
        "Pani Puri, also known as Golgappa, is a popular street snack that consists of a round, hollow puri filled with a mixture of flavored water (pani), tamarind chutney, chili, chaat masala, potato, onion, and chickpeas.",
        "Palak Paneer is a North Indian curry made from spinach and paneer (Indian cottage cheese) in a thick curry sauce based on pureed spinach, seasoned with garlic, garam masala, and other spices.",
        "Aloo Gobi is a vegetarian dish from the Indian subcontinent made with potatoes (aloo), cauliflower (gobi), and Indian spices. It is yellowish in color due to the use of turmeric, and sometimes contains kalonji and curry leaves.",
        "Samosas are a popular teatime snack and are often served as an appetizer. They are deep-fried pastries with a savory filling, such as spiced potatoes, onions, peas, and lentils.",
        "Butter Chicken is a rich and creamy North Indian curry made with marinated chicken cooked in a tomato-based sauce with butter, cream, and a blend of spices. It is often served with naan or rice.",
        "Chole Bhature is a combination of spicy chickpea curry and deep-fried bread called bhature. This North Indian dish is often enjoyed for breakfast or lunch and is served with onions, pickles, and yogurt.",
        "Rasam is a South Indian soup made from tamarind juice, tomatoes, and a blend of spices including mustard seeds, curry leaves, and black pepper. It is typically served with rice.",
        "Dhokla is a steamed savory cake made from fermented rice and chickpea batter. It is a popular snack from Gujarat and is often garnished with mustard seeds, green chilies, and coriander.",
        "Pav Bhaji is a Mumbai street food consisting of a spicy vegetable mash (bhaji) served with buttered bread rolls (pav). It is garnished with chopped onions, cilantro, and a squeeze of lemon.",
        "Vindaloo is a Goan curry known for its fiery heat and tangy flavor. It is made with marinated pork cooked in a sauce of vinegar, garlic, and a variety of spices including red chilies and cinnamon.",
        "Dum Aloo is a popular North Indian dish where potatoes are cooked slowly in a rich, spiced yogurt gravy. It is often garnished with coriander and served with rice or bread.",
        "Idli is a South Indian breakfast dish made from steamed fermented rice and lentil batter. It is typically served with coconut chutney and sambar."
    ]

    # Store each document in a vector embedding database
    for i, d in enumerate(documents):
        response = ollama.embeddings(model=embedding_model, prompt=d)
        embedding = response["embedding"]
        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[d]
        )

# Function to search items with the query
def language_chat(message, history):
    response = ollama.embeddings(prompt=message, model=embedding_model)
    results = collection.query(
        query_embeddings=[response["embedding"]], n_results=2)
    
    # Shorten and format the results
    shortened_results = [doc.split(".")[0] for doc in results['documents'][0]]
    return "\n".join(shortened_results)

# Gradio Chat Interface
demo = gr.ChatInterface(
    language_chat, title="Vector DB Search", theme='taithrah/minimal'
)

if __name__ == "__main__":
    demo.launch()
