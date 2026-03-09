"""
Author: Preethi Gajawada
Description: Non-confrontational tone reward using emotion classification.
"""

# dspy_cn/rewards/r4_non_confrontational.py
# R4: Non-Confrontational Tone — maps to PRS
# Model: j-hartmann/emotion-english-distilroberta-base
from transformers import pipeline
from .base import RewardInput


class NonConfrontationalTone:
    def __init__(self):
        self.emotion = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )

    def score(self, inp: RewardInput) -> float:
        preds = self.emotion(inp.counter_narrative)
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        neg = sum(p["score"] for p in preds if p["label"] in {"anger", "disgust"})
        return float(max(0.0, 1.0 - neg))
