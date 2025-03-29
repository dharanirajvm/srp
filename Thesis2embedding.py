# Install required libraries
#To run in GColab as the dataset is in drive

import os
import fitz  # PyMuPDF
import nltk
import spacy
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


nltk.download('punkt_tab')
stop_words = set(stopwords.words("english"))
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load English NLP model
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words("english"))

def extract_text_from_pdfs(folder_path):
    """Extracts text from PDFs and TXT files in a folder."""
    documents = {}
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            documents[file] = text
        elif file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as txt_file:
                documents[file] = txt_file.read()
    return documents

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    doc = nlp(" ".join(filtered_tokens))
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define folder path
folder_path = "/content/drive/MyDrive/Project Phase I Report"  # Update with your dataset folder

# Extract and preprocess text from all research papers
research_papers = extract_text_from_pdfs(folder_path)
processed_papers = {file: preprocess_text(text) for file, text in research_papers.items()}

# Convert processed text into embeddings
paper_embeddings = [model.encode(text, convert_to_numpy=True) for text in processed_papers.values()]
paper_files = list(processed_papers.keys())

# Initialize FAISS index
embedding_dim = paper_embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)  # L2 Distance Index

# Convert embeddings to NumPy array and store in FAISS
corpus_embeddings = np.array(paper_embeddings)
index.add(corpus_embeddings)

# Save FAISS index and file names for future use
faiss.write_index(index, "faiss_index.bin")
np.save("paper_files.npy", paper_files)

print(f"FAISS index saved successfully with {len(paper_files)} documents!")
