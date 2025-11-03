# src/config.py
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# model names
SENT_MODEL_NAME = "distilroberta-base"  # base checkpoint to LoRA-finetune
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# saved artifacts
SENT_MODEL_PATH = os.path.join(MODELS_DIR, "lora_sentiment")
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "faiss_index.bin")
EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "review_embeddings.npy")
REVIEWS_PICKLE = os.path.join(MODELS_DIR, "reviews_df.pkl")
