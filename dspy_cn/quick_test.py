# dspy_cn/quick_test.py
# Quick sanity check: model loading, generation, judge, and reward functions.
import os
import yaml
from dspy_cn.base_llm import configure_llama_local
from dspy_cn.dspy_program import EnglishCNProgram
from dspy_cn.dspy_metric import llm_judge


def main():
    with open("dspy_cn/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg["base_llm"]

    print("[1/5] Loading Llama 3.1-8B locally...")
    configure_llama_local(
        model_id=llm_cfg["model_id"],
        max_tokens=llm_cfg["max_tokens"],
        temperature=llm_cfg["temperature"],
    )
    print("[OK] Model loaded.\n")

    # ── English test ──
    print("[2/5] Testing English generation (Preethi baseline prompt)...")
    en_prog = EnglishCNProgram()
    test_hs = "Being gay is unnatural and against god's plan."
    pred = en_prog(hate_speech=test_hs)
    cn = pred.counter_narrative
    print(f"  HS:  {test_hs}")
    print(f"  CN:  {cn}")
    print(f"  Len: {len(cn.split())} words\n")

    # ── LLM Judge ──
    print("[3/5] Testing LLM-as-judge...")
    if os.environ.get("OPENAI_API_KEY"):
        scores = llm_judge(test_hs, cn)
        print(f"  Judge: PRS={scores['PRS']}, CCNC={scores['CCNC']}, QS={scores['QS']} ({scores['pct']:.1f}%)\n")
    else:
        print("  [SKIP] OPENAI_API_KEY not set\n")

    # ── Reward Functions ──
    print("[4/5] Testing reward functions...")
    try:
        from dspy_cn.evaluator import RewardEvaluator
        evaluator = RewardEvaluator()
        gt = "Being gay is completely natural and part of human diversity."
        reward_scores = evaluator.score_single(test_hs, cn, gt)
        print(f"  R1 (Safety):       {reward_scores['R1']:.3f}")
        print(f"  R2 (Empathy):      {reward_scores['R2']:.3f}")
        print(f"  R3 (Grounding):    {reward_scores['R3']:.3f}")
        print(f"  R4 (Tone):         {reward_scores['R4']:.3f}")
        print(f"  R8 (Contradiction):{reward_scores['R8']:.3f}")
        print(f"  R12 (GT Cosine):   {reward_scores['R12']:.3f}")
        print(f"  PRS (composite):   {reward_scores['PRS']:.3f}")
        print(f"  CCNC (composite):  {reward_scores['CCNC']:.3f}")
        print(f"  TS (composite):    {reward_scores['TS']:.3f}")
        print(f"  Distinct-2:        {reward_scores['Distinct2']:.3f}\n")
    except Exception as e:
        print(f"  [WARN] Reward functions failed: {e}")
        print("  Install missing packages: pip install --user detoxify sentence-transformers bert-score\n")

    # ── BERTScore ──
    print("[5/5] Testing BERTScore...")
    try:
        from dspy_cn.rewards import BERTScoreMetric
        from dspy_cn.rewards.base import RewardInput
        bs = BERTScoreMetric()
        inp = RewardInput(hate_speech=test_hs, counter_narrative=cn, ground_truth=gt)
        bs_score = bs.score(inp)
        print(f"  BERTScore F1: {bs_score:.4f}\n")
    except Exception as e:
        print(f"  [WARN] BERTScore failed: {e}\n")

    print("[DONE] All checks complete!")


if __name__ == "__main__":
    main()
