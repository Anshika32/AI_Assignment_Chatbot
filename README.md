# Chatbot

This project is a web application for a chatbot that answers questions related to a Jessup Cellars Corpus. It uses machine learning to understand and retrieve relevant information from a set of provided documents. The application is built using Streamlit and various NLP libraries to provide accurate and concise responses.

📋 **Table of Contents**
- [Features](#features)
- [File Descriptions](#file-descriptions)
- [Acknowledgements](#acknowledgements)
- [How to Use](#how-to-use)

✨ **Features**
- Extracts and processes text from PDF documents.
- Answers questions based on a predefined corpus.
- Provides concise and relevant responses.
- Simple and intuitive user interface.
- Maintains conversation history.

📂 **File Descriptions**
- `Data_Preparation.ipynb`: Google Colab notebook for preparing data, extracting text from PDFs, and generating embeddings.
- `app.py`: The main Streamlit application file that handles the user interface and chatbot logic.
- `texts_and_embeddings.pkl`: Pickle file containing processed text and embeddings generated from the corpus.
- `requirements.txt`: List of Python packages required to run the application.

🙏 **Acknowledgements**
- The project uses Streamlit for building the web application interface.
- Text processing and embedding generation are done using LangChain, PyPDF2, FAISS, and Transformers libraries.

### How to Use

1. **Clone the Repository**
   - Clone the repository containing the code to your local machine:
     ```bash
     git clone https://github.com/Anshika32/AI_Assignment_Chatbot.git
     ```

2. **Install Required Libraries**
   - Open a terminal and navigate to the directory where the repository is cloned.
   - Install the required libraries using pip:
     ```bash
     pip install -r requirements.txt
     ```

3. **Google Colab (Data Preparation)**
   - Open the `Data_Preparation.ipynb` notebook in Google Colab.
   - Run all the cells in the notebook to generate the `texts_and_embeddings.pkl` file.
   - This notebook will:
     - Mount Google Drive.
     - Read and process the PDF file.
     - Generate embeddings for text chunks.
     - Save the texts and embeddings into a pickle file.

4. **Transfer the Pickle File**
   - Download the `texts_and_embeddings.pkl` file from Google Drive and place it in the directory where your Streamlit app code is located.

5. **Run the Streamlit Interface**
   - Open a terminal and navigate to the directory containing the Streamlit code.
   - Run the Streamlit app using the command:
     ```bash
     streamlit run app.py
     ```
   - This will start a local server. Open the provided URL in your web browser to interact with the chatbot.

By following these steps, you can set up and run the wine business chatbot on your local system. The chatbot provides a simple interface for users to ask questions and get relevant information from the provided corpus, making it a useful tool for customer engagement and support.
