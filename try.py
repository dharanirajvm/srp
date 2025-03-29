from sentence_transformers import SentenceTransformer
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Model is cached at: {model.cache_folder}")