# src/eval.py
import pandas as pd
from models_utils import load_sentiment_pipeline
from config import REVIEWS_PICKLE

def test_sentiment_accuracy(sample_n=200):
    df = pd.read_pickle(REVIEWS_PICKLE)
    df = df[df['review'].str.strip() != ""]
    df = df.sample(min(sample_n, len(df)))
    sentiment = load_sentiment_pipeline()
    correct = 0
    total = 0
    for _, row in df.iterrows():
        true_star = row.get('overall_star', '')
        if true_star=="":
            continue
        try:
            true_label = "POSITIVE" if float(true_star)>=4 else ("NEGATIVE" if float(true_star)<=2 else "NEUTRAL")
        except:
            true_label = "NEUTRAL"
        out = sentiment(row['review'][:512])[0]
        pred = out['label'].upper()
        if "POS" in pred and "POS" in true_label:
            correct += 1
        elif "NEG" in pred and "NEG" in true_label:
            correct += 1
        elif "NEUT" in pred and "NEUT" in true_label:
            correct += 1
        total += 1
    print("Accuracy:", correct/total if total>0 else None)
