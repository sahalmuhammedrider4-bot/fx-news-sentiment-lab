"""Headline sentiment scoring. VADER by default, FinBERT optionally."""
from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# VADER is built for social media, so we nudge it with market vocabulary.
FINANCE_LEXICON = {
    "surge": 2.0, "surges": 2.0, "rally": 2.0, "rallies": 2.0, "jumps": 1.8,
    "gains": 1.5, "strengthens": 1.8, "upgrade": 1.8, "upbeat": 1.8,
    "plunge": -2.5, "plunges": -2.5, "slump": -2.3, "slides": -1.8, "tumbles": -2.3,
    "weakens": -1.8, "selloff": -2.2, "recession": -2.5, "crisis": -2.5,
    "default": -2.3, "downgrade": -1.8, "losses": -1.5, "inflation": -0.8,
    "hawkish": -0.5, "dovish": 0.5,
}


def score_vader(titles: list[str]) -> list[float]:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(FINANCE_LEXICON)
    return [sia.polarity_scores(t)["compound"] for t in titles]


def score_finbert(titles: list[str]) -> list[float]:
    """Finance-tuned transformer. Needs: pip install transformers torch (downloads ~440MB once)."""
    from transformers import pipeline

    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
    sign = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    out = clf(list(titles), batch_size=32)
    return [sign[o["label"].lower()] * o["score"] for o in out]


def add_sentiment(headlines: pd.DataFrame, backend: str = "vader") -> pd.DataFrame:
    scorer = score_finbert if backend == "finbert" else score_vader
    out = headlines.copy()
    out["sentiment"] = scorer(out["title"].tolist())
    return out
