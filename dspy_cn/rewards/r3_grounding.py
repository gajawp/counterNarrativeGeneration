"""
Author: Preethi Gajawada
Description: Semantic grounding reward using sentence-transformers.
"""

# dspy_cn/rewards/r3_grounding.py
# R3: Semantic Grounding — maps to CCNC
# Model: sentence-transformers/all-MiniLM-L6-v2
from sentence_transformers import SentenceTransformer, util
from .base import RewardInput


class InputOutputSemanticGrounding:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def score(self, inp: RewardInput) -> float:
        a = self.model.encode(inp.hate_speech, convert_to_tensor=True)
        b = self.model.encode(inp.counter_narrative, convert_to_tensor=True)
        return float(util.cos_sim(a, b).item())
