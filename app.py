import os
import streamlit as st
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartForConditionalGeneration
import faiss
from collections import deque

# Setting environment variable to avoid OpenMP runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the pre-trained model from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load the summarization model
summarization_model_name = "facebook/bart-large-cnn"
summarizer_tokenizer = BartTokenizer.from_pretrained(summarization_model_name)
summarizer_model = BartForConditionalGeneration.from_pretrained(summarization_model_name)

# Load the texts and embeddings from the pickle file
with open('texts_and_embeddings.pkl', 'rb') as f:
    texts, embeddings_matrix = pickle.load(f)

# Create FAISS index and add embeddings
d = embeddings_matrix.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings_matrix)

# Function to get embeddings using the Hugging Face model
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to search for similar documents
def similarity_search(query, k=4):
    query_embedding = get_embeddings(query).cpu().numpy().astype('float32')
    D, I = index.search(query_embedding, k)
    return [texts[i] for i in I[0]]

# Function to summarize the text
def summarize_text(text):
    inputs = summarizer_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer_model.generate(
        inputs.input_ids, 
        max_length=400,
        min_length=100,
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Initialize conversation history
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=10)

# Function to handle conversation history
def update_history(history, user_input, response):
    history.append({"user": user_input, "bot": response})
    return history

# Streamlit application
st.title("Chatbot")
st.write("Ask a question get relevant information.")

query = st.text_input("Enter your question:")

if query:
    docs = similarity_search(query, k=10)  # Increase k to get more documents
    combined_docs = " ".join(docs)
    summary = summarize_text(combined_docs)
    
    # Check for out-of-corpus questions
    if not summary.strip():
        summary = "I'm sorry, I don't have the information you're looking for. Please contact our business directly for more details."

    # Update conversation history
    st.session_state.history = update_history(st.session_state.history, query, summary)
    
    # Display conversation
    for turn in st.session_state.history:
        st.write(f"**You:** {turn['user']}")
        st.write(f"**Bot:** {turn['bot']}")

# Sidebar for conversation history
st.sidebar.title("Conversation History")
for turn in st.session_state.history:
    st.sidebar.write(f"**You:** {turn['user']}")
    st.sidebar.write(f"**Bot:** {turn['bot']}")
