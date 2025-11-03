"""Utility functions that orchestrate retrieval, sentiment, and LLM generation.

Uses existing `models_utils` for retrieval and sentiment, and `lora_agent` to
load a PEFT LoRA adapter for a causal LLM. The AgentOrchestrator exposes a
single method `generate_course_advice(summary, query)` which returns a string.
"""

from typing import List, Dict
import logging

from models_utils import search_reviews, load_sentiment_pipeline
from lora_agent import make_default_agent

LOGGER = logging.getLogger(__name__)


def build_prompt(summary: Dict, top_reviews: List[str]) -> str:
    """Create a prompt combining course summary and top reviews for the LLM."""
    header = (
        f"You are a helpful course advisor assistant. Provide a concise recommendation "
        f"and summary for the course {summary.get('course')} taught by {summary.get('prof')}."
    )
    bullets = [
        f"- Average Grade: {summary.get('avg_grade')}", f"- Average Rating: {summary.get('avg_star')}"]
    reviews_section = "\n\nTop student reviews:\n"
    for i, r in enumerate(top_reviews[:5], 1):
        reviews_section += f"{i}. {r}\n"

    prompt = (
        header
        + "\n"
        + "\n".join(bullets)
        + reviews_section
        + "\nProvide:\n1) A short summary (2-3 sentences)\n2) Key pros and cons from reviews\n3) A final recommendation for a student deciding whether to take this course."
    )
    return prompt


class AgentOrchestrator:
    def __init__(self, adapter_dir: str = None):
        self.adapter_dir = adapter_dir
        self.agent = make_default_agent(adapter_dir=adapter_dir)
        # sentiment pipeline is optional; fallback to None if missing
        try:
            self.sentiment = load_sentiment_pipeline()
        except Exception:
            LOGGER.exception("Failed to load sentiment pipeline")
            self.sentiment = None

    def generate_course_advice(self, summary: Dict, query: str) -> str:
        # Fetch top reviews via semantic search
        try:
            df = search_reviews(query, top_k=5)
            top_reviews = df['review'].dropna().tolist()
        except Exception:
            LOGGER.exception("Failed review search")
            top_reviews = []

        # Optionally annotate sentiments
        pros: List[str] = []
        cons: List[str] = []
        if self.sentiment and top_reviews:
            try:
                preds = self.sentiment([r for r in top_reviews])
                for r, p in zip(top_reviews, preds):
                    # pipeline returns label like POSITIVE/NEGATIVE or a dict-like output
                    label = p.get('label') if isinstance(p, dict) else str(p)
                    if isinstance(label, str) and label.lower().startswith('pos'):
                        pros.append(r)
                    elif isinstance(label, str) and label.lower().startswith('neg'):
                        cons.append(r)
            except Exception:
                LOGGER.exception("Sentiment analysis failed")

        # Build prompt and call LLM
        prompt = build_prompt(summary, top_reviews)
        raw = self.agent.generate(prompt)

        # Post-process: attach extracted pros/cons if LLM returned nothing
        if not raw:
            parts = ["Summary:"]
            if pros:
                parts.append("Pros:\n" + "; ".join(pros[:3]))
            if cons:
                parts.append("Cons:\n" + "; ".join(cons[:3]))
            parts.append(
                "Recommendation: Use summary and review sentiments to decide.")
            raw = "\n\n".join(parts)

        return raw


__all__ = ["AgentOrchestrator"]
