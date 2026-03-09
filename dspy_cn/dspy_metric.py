# dspy_cn/dspy_metric.py
# Two metric modes:
#   1. LLM-as-judge (GPT-4o-mini with shared task rubric)
#   2. Reward-based (using subset reward functions for PRS/CCNC/TS)
# Both return float 0-1 for DSPy optimizers.
import json
import os
import traceback
from openai import OpenAI

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


# ── LLM-as-Judge ───────────────────────────────────────────────

PRS_RUBRIC = """Criterion: Does the generated counter-narrative reflect a polite and respectful tone relevant to LGBTQ+ individuals and communities?
0 = No polite or respectful language used in the counter-narrative framing
1 = Attempts polite and respectful framing, but the tone is inconsistent
2 = Clearly and consistently reflects a polite and respectful tone"""

CCNC_RUBRIC = """Criterion: Does the counter-narrative remain coherent, relevant, and contextually appropriate in response to the specific harmful or offensive speech?
0 = The counter-narrative is off-topic, incoherent, or fails to address the context
1 = Partially coherent or contextually relevant but lacks clarity or consistency
2 = Clearly and consistently coherent, relevant, and context-aware throughout the response"""

QS_RUBRIC = """Criterion: Does the counter-narrative demonstrate richness, grammatical correctness, and strong relatedness to the harmful speech?
0 = Poor quality with little depth, major grammatical issues, or weak/no connection to the harmful speech
1 = Some depth and generally understandable, but may contain minor grammatical issues or partial relatedness
2 = Clear, well-written, rich in content, and strongly related to the harmful speech"""


def llm_judge(hate_speech: str, counter_narrative: str,
              model: str = "gpt-4o-mini") -> dict:
    """Score a single CN against the 3 shared task rubrics."""
    prompt = f"""You are an impartial judge. Score the counter-narrative on THREE criteria.

HATE SPEECH:
{hate_speech}

COUNTER-NARRATIVE:
{counter_narrative}

--- CRITERION 1: Politeness and Respectful Score (PRS) ---
{PRS_RUBRIC}

--- CRITERION 2: Contextual Counter-Narrative Coherence Score (CCNC) ---
{CCNC_RUBRIC}

--- CRITERION 3: Quality Score (QS) ---
{QS_RUBRIC}

Return ONLY valid JSON with integer scores:
{{"PRS": <0-2>, "CCNC": <0-2>, "QS": <0-2>}}"""

    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=50,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores = json.loads(content)
        prs = int(scores.get("PRS", 0))
        ccnc = int(scores.get("CCNC", 0))
        qs = int(scores.get("QS", 0))
    except Exception as e:
        print(f"[WARN] Judge parse error: {e}")
        prs, ccnc, qs = 1, 1, 1

    total = prs + ccnc + qs
    return {"PRS": prs, "CCNC": ccnc, "QS": qs, "total": total, "pct": total / 6 * 100}


# ── DSPy Metric Functions ──────────────────────────────────────

def cn_metric_llm_judge(example, prediction, trace=None) -> float:
    """DSPy metric using LLM-as-judge. Returns 0-1."""
    hs = str(example.hate_speech)
    cn = str(prediction.counter_narrative)
    scores = llm_judge(hs, cn)
    return scores["total"] / 6.0


def cn_metric_rewards(example, prediction, trace=None) -> float:
    """DSPy metric using reward functions. Returns 0-1.
    Lazily loads the evaluator on first call."""
    global _reward_evaluator
    if "_reward_evaluator" not in globals() or _reward_evaluator is None:
        from dspy_cn.evaluator import RewardEvaluator
        globals()["_reward_evaluator"] = RewardEvaluator()

    hs = str(example.hate_speech)
    cn = str(prediction.counter_narrative)
    gt = str(example.ground_truth) if hasattr(example, "ground_truth") else None

    scores = _reward_evaluator.score_single(hs, cn, gt)
    return float(scores["combined"])
