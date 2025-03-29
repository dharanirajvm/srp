import fitz
import faiss
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer

nltk.download('punkt_tab')

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained model and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(nltk.corpus.stopwords.words("english"))

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

# Load FAISS index and document metadata
index = faiss.read_index("faiss_index.bin")
paper_files = np.load("paper_files.npy", allow_pickle=True)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

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
        similarity_score = 1 - distances[0][i]  # Convert L2 distance to similarity score
        print(f"{i+1}. {paper_files[indices[0][i]]} (Score: {similarity_score:.4f})")

# Example Usage
search_similar_papers("2023176001_Mouliraj_A_K.pdf", top_k=3)