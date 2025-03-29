import fitz
import faiss
import numpy as np
import pandas as pd
import nltk
import spacy
from sentence_transformers import SentenceTransformer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words("english"))

# Load metadata from CSV file
metadata_df = pd.read_csv("metadata.csv")
metadata_dict = metadata_df.set_index("filename").to_dict(orient="index")

# Load FAISS index and document filenames
index = faiss.read_index("faiss_index.bin")
paper_files = np.load("paper_files.npy", allow_pickle=True)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

def preprocess_text(text):
    """Tokenizes, removes stopwords, and lemmatizes text."""
    tokens = nltk.word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    doc = nlp(" ".join(filtered_tokens))
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    
    return lemmatized_text

def search_similar_papers(pdf_path, top_k=3):
    """Finds the top-k most similar research papers to a given thesis document."""
    # Extract and preprocess the input PDF
    thesis_text = extract_text_from_pdf(pdf_path)
    processed_thesis = preprocess_text(thesis_text)

    # Convert the input text to an embedding
    query_embedding = model.encode(processed_thesis, convert_to_numpy=True).reshape(1, -1)

    # Perform FAISS search
    distances, indices = index.search(query_embedding, top_k)

    print("\nTop similar research papers:")
    for i in range(top_k):
        filename = paper_files[indices[0][i]]
        similarity_score = 1 - distances[0][i]  # Convert L2 distance to similarity score
        
        # Retrieve metadata
        metadata = metadata_dict.get(filename, {})
        author = metadata.get("author", "Unknown")
        title = metadata.get("title", "Unknown")
        year = metadata.get("year", "Unknown")
        institute = metadata.get("institute", "Unknown")
        link = metadata.get("link", "Unavailable")
        
        print(f"{i+1}. {title} ({year})")
        print(f"   Author: {author}")
        print(f"   Institute: {institute}")
        print(f"   Similarity Score: {similarity_score:.4f}")
        print(f"   Link: {link}\n")

# Example Usage
search_similar_papers("plagiarism_detection-data/data/g0pA_taskd.txt", top_k=3)
