"""
Author: Preethi Gajawada
Description: Empathy reward based on emotion classification.
"""

# dspy_cn/rewards/r2_empathy.py
# R2: Empathy — maps to PRS (Politeness and Respectful Score)
# Model: j-hartmann/emotion-english-distilroberta-base
from transformers import pipeline
from .base import RewardInput


def _clamp01(x):
    return float(max(0.0, min(1.0, x)))


class EmpathyReward:
    def __init__(self):
        self.clf = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None,
        )

    def score(self, inp: RewardInput) -> float:
        preds = self.clf(inp.counter_narrative)
        if preds and isinstance(preds[0], list):
            preds = preds[0]
        emotion_map = {p["label"]: p["score"] for p in preds}
        empathy = (
            emotion_map.get("joy", 0.0)
            + emotion_map.get("neutral", 0.0)
        ) - (
            emotion_map.get("anger", 0.0)
            + emotion_map.get("disgust", 0.0)
        )
        return _clamp01(0.5 + empathy)
