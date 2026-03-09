"""
Author: Preethi Gajawada
Institution: Northeastern University
Project: Multilingual Counter-Narrative Generation using DSPy
Description: Generates counter-narratives for test datasets and evaluates them using LLM judge and reward metrics.
Year: 2026
"""

# dspy_cn/generate_predictions.py
# Generate CNs for test sets, evaluate with BOTH LLM judge AND reward functions.
# Outputs include: PRS, CCNC, QS, Distinct-2, BERTScore, all individual rewards.
import os
import yaml
import json
import pandas as pd
from tqdm import tqdm
from dspy_cn.base_llm import configure_gpt4o
from dspy_cn.dspy_program import (
    EnglishCNProgram, TamilCNProgram,
    EnglishCNSignature, TamilCNSignature,
)
from dspy_cn.dspy_metric import llm_judge
from dspy_cn.evaluator import RewardEvaluator


def generate_for_dataset(program, df, hs_col, id_col, evaluator,
                         judge_model="gpt-4o-mini"):
    """Generate CN for each row, score with LLM judge + reward functions."""
    results = []
    cn_list, hs_list, gt_list = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        hs = str(row[hs_col])
        row_id = str(row[id_col])
        gt = str(row.get("ground truth", "")) if "ground truth" in df.columns else None

        try:
            pred = program(hate_speech=hs)
            cn = str(pred.counter_narrative)
        except Exception as e:
            print(f"[ERR] {row_id}: {e}")
            cn = "Every person deserves dignity and respect regardless of identity."

        # LLM Judge scores
        judge_scores = llm_judge(hs, cn, model=judge_model)

        # Reward function scores
        reward_scores = evaluator.score_single(hs, cn, gt)

        results.append({
            "Id": row_id,
            "hate_speech": hs,
            "generated_cn": cn,
            "ground_truth": gt or "",
            # LLM Judge
            "judge_PRS": judge_scores["PRS"],
            "judge_CCNC": judge_scores["CCNC"],
            "judge_QS": judge_scores["QS"],
            "judge_pct": judge_scores["pct"],
            # Reward functions
            "reward_PRS": reward_scores["PRS"],
            "reward_CCNC": reward_scores["CCNC"],
            "reward_QS": reward_scores["QS"],
            "reward_combined": reward_scores["combined"],
            # Distinct-2 (per-item)
            "Distinct2": reward_scores["Distinct2"],
        })

        cn_list.append(cn)
        hs_list.append(hs)
        gt_list.append(gt or "")

    df_results = pd.DataFrame(results)

    # Batch-level metrics
    batch_d2 = evaluator.distinct2.score_batch(cn_list)
    valid_gts = [g for g in gt_list if g]
    if valid_gts and len(valid_gts) == len(cn_list):
        bs_result = evaluator.bertscore.score_batch(cn_list, gt_list)
        df_results["BERTScore_F1"] = bs_result["per_item"]
        avg_bertscore = bs_result["avg_f1"]
    else:
        df_results["BERTScore_F1"] = 0.0
        avg_bertscore = 0.0

    return df_results, batch_d2, avg_bertscore


def print_summary(df, label, batch_d2, avg_bs):
    n = len(df)
    print(f"\n{'='*60}")
    print(f"  {label} — {n} examples")
    print(f"{'='*60}")
    print(f"  --- LLM Judge (GPT-4o-mini) ---")
    print(f"  PRS  avg: {df['judge_PRS'].mean():.3f} / 2")
    print(f"  CCNC avg: {df['judge_CCNC'].mean():.3f} / 2")
    print(f"  QS   avg: {df['judge_QS'].mean():.3f} / 2")
    print(f"  Overall:  {df['judge_pct'].mean():.1f}%")
    perfect = len(df[(df['judge_PRS']==2) & (df['judge_CCNC']==2) & (df['judge_QS']==2)])
    print(f"  Perfect:  {perfect}/{n}")
    print(f"  --- Reward Functions ---")
    print(f"  PRS  (R2+R4+R1): {df['reward_PRS'].mean():.3f}")
    print(f"  CCNC (R3+R8+R12): {df['reward_CCNC'].mean():.3f}")
    print(f"  QS   (R1+R4):    {df['reward_QS'].mean():.3f}")
    print(f"  Combined:        {df['reward_combined'].mean():.3f}")
    print(f"  --- Reference Metrics ---")
    print(f"  Distinct-2 (batch): {batch_d2:.4f}")
    print(f"  BERTScore F1 (avg): {avg_bs:.4f}")
    print(f"{'='*60}")


def main():
    with open("dspy_cn/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    llm_cfg = cfg["base_llm"]

    judge_model = cfg["judge_llm"]["model"]

    configure_gpt4o(model=llm_cfg["model"])

    evaluator = RewardEvaluator()
    os.makedirs("dspy_cn/outputs", exist_ok=True)

    # # ── English ──
    print("\n[1/2] Generating English counter-narratives...")
    test_en = pd.read_csv(cfg["data"]["test_en_csv"])

    # Try loading optimized program
    en_program = EnglishCNProgram()
    for opt_path in [
                     "dspy_cn/outputs/en_copro_optimized.json"]:
        if os.path.exists(opt_path):
            print(f"[OK] Loading optimized program: {opt_path}")
            try:
                en_program.load(opt_path)
                break
            except Exception as e:
                print(f"[WARN] Could not load {opt_path}: {e}")

    df_en, d2_en, bs_en = generate_for_dataset(
        en_program, test_en, "text", "Id", evaluator, judge_model)
    df_en.to_csv("dspy_cn/outputs/predictions_en.csv", index=False)
    print_summary(df_en, "ENGLISH", d2_en, bs_en)

    # ── Tamil ──
    print("\n[2/2] Generating Tamil counter-narratives...")
    test_ta = pd.read_csv(cfg["data"]["test_ta_csv"])

    ta_program = TamilCNProgram()
    for opt_path in [
                     "dspy_cn/outputs/ta_copro_optimized.json"]:
        if os.path.exists(opt_path):
            print(f"[OK] Loading optimized program: {opt_path}")
            try:
                ta_program.load(opt_path)
                break
            except Exception as e:
                print(f"[WARN] Could not load {opt_path}: {e}")

    df_ta, d2_ta, bs_ta = generate_for_dataset(
        ta_program, test_ta, "text", "Id", evaluator, judge_model)
    df_ta.to_csv("dspy_cn/outputs/predictions_ta.csv", index=False)
    print_summary(df_ta, "TAMIL", d2_ta, bs_ta)

    # ── Submission CSVs ──
    sub_en = df_en[["Id", "generated_cn"]].rename(columns={"generated_cn": "counter_narrative"})
    sub_en.to_csv("dspy_cn/outputs/submission_en.csv", index=False)
    sub_ta = df_ta[["Id", "generated_cn"]].rename(columns={"generated_cn": "counter_narrative"})
    sub_ta.to_csv("dspy_cn/outputs/submission_ta.csv", index=False)

    print("\n[OK] All files saved to dspy_cn/outputs/")


if __name__ == "__main__":
    main()
