from config import SENT_MODEL_PATH, EMBED_MODEL, FAISS_INDEX_PATH, REVIEWS_PICKLE, EMBEDDINGS_PATH
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import os
import re

# ===============================
# LOADERS
# ===============================

_sentiment_pipe = None
_embedder = None
_faiss_index = None
_reviews_df = None

def load_sentiment_pipeline():
    """Loads the sentiment analysis pipeline with fallback."""
    global _sentiment_pipe

    if _sentiment_pipe is None:
        try:
            if os.path.exists(SENT_MODEL_PATH):
                tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL_PATH)
                model = AutoModelForSequenceClassification.from_pretrained(
                    SENT_MODEL_PATH,
                    ignore_mismatched_sizes=True,
                    dtype="auto"
                )
                _sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            else:
                _sentiment_pipe = pipeline("sentiment-analysis")
        except Exception as e:
            print(f"⚠️ Failed to load sentiment pipeline from {SENT_MODEL_PATH}: {e}")
            print("→ Falling back to default sentiment-analysis model.")
            _sentiment_pipe = pipeline("sentiment-analysis")

    return _sentiment_pipe


def load_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_faiss():
    global _faiss_index, _reviews_df
    if _faiss_index is None:
        if os.path.exists(FAISS_INDEX_PATH):
            _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        else:
            raise FileNotFoundError("Faiss index not found: please run build_index.py")

    if _reviews_df is None:
        _reviews_df = pd.read_pickle(REVIEWS_PICKLE)
    return _faiss_index, _reviews_df


def search_reviews(query, top_k=5):
    emb = load_embedder().encode([query])
    idx, reviews = load_faiss()
    D, I = idx.search(emb.astype('float32'), top_k)
    df = reviews.iloc[I[0]].copy()
    df['dist'] = D[0].tolist()
    return df

# ===============================
# COURSE ADVISOR AGENT
# ===============================

COURSE_STATS_PATH = "data/course_stats.csv"
course_stats = pd.read_csv(COURSE_STATS_PATH)

# ---------- Intent Detection ----------
def detect_intent(query: str):
    query = query.lower().strip()
    course_match = re.findall(r"[A-Z]{2,3}\d{3}[A-Z]?", query.upper())

    if course_match:
        return {"intent": "course_query", "course_code": course_match[0]}

    if "how many" in query and any(g in query for g in ["a", "b", "c", "grade"]):
        return {"intent": "grade_count"}

    if any(k in query for k in ["good grading", "best grading", "easy grading", "a centric", "high average", "average above", "average over", "average grading of"]):
        match = re.search(r"(\d+(\.\d+)?)", query)
        threshold = float(match.group(1)) if match else 8.0
        return {"intent": "high_grade_courses", "threshold": threshold}

    if any(k in query for k in ["ai", "ml", "machine learning", "artificial intelligence", "data science"]):
        return {"intent": "ai_ml_courses"}

    return {"intent": "unknown"}

# ---------- Main Agent Response ----------
def agent_response(query):
    intent = detect_intent(query)

    if intent["intent"] == "course_query":
        code = intent["course_code"]
        row = course_stats[course_stats["course_code"].str.upper() == code.upper()]
        if row.empty:
            return {"text": f"🤔 No data found for {code}.", "image": None}
        avg = float(row["avg_grade"].iloc[0])
        prof = row["prof_name"].iloc[0]
        samples = int(row["samples"].iloc[0])
        return {"text": f"📘 {code} — {prof}\nAverage Grade: {avg}\nSamples: {samples}", "image": None}

    elif intent["intent"] == "high_grade_courses":
        threshold = intent.get("threshold", 8.0)
        good = course_stats[course_stats["avg_grade"] >= threshold]
        if good.empty:
            return {"text": f"😕 No courses found with average grade ≥ {threshold}.", "image": None}
        text = f"🏆 Courses with average grade ≥ {threshold}:\n"
        for _, row in good.sort_values("avg_grade", ascending=False).head(10).iterrows():
            text += f"- {row['course_code']} ({row['prof_name']}): {row['avg_grade']:.2f}\n"
        return {"text": text.strip(), "image": None}

    elif intent["intent"] == "ai_ml_courses":
        ai = course_stats[course_stats["course_name"].str.contains("AI|ML|Machine Learning|Deep Learning|Data Science", case=False, na=False)]
        if ai.empty:
            return {"text": "🤖 No AI/ML-related courses found.", "image": None}
        text = "💡 AI/ML-related courses with good grading:\n"
        for _, row in ai.sort_values("avg_grade", ascending=False).head(10).iterrows():
            text += f"- {row['course_code']} ({row['prof_name']}): {row['avg_grade']:.2f}\n"
        return {"text": text.strip(), "image": None}

    elif intent["intent"] == "grade_count":
        return {"text": "📈 Please specify the course code, e.g. 'How many students got B+ in MSE303'", "image": None}

    else:
        return {"text": "🤖 Try asking: 'Courses with average ≥ 8' or 'AI/ML courses with good grading'.", "image": None}
