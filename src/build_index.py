# src/build_index.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
from config import EMBED_MODEL, MODELS_DIR, FAISS_INDEX_PATH, EMBEDDINGS_PATH, REVIEWS_PICKLE

def main():
    reviews = pd.read_pickle(REVIEWS_PICKLE)
    texts = reviews['review'].tolist()
    embed = SentenceTransformer(EMBED_MODEL)
    embeddings = embed.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # save embeddings & reviews
    np.save(EMBEDDINGS_PATH, embeddings)
    reviews.to_pickle(REVIEWS_PICKLE)

    # build faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Built FAISS index and saved to", FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()
