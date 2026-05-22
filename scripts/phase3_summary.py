"""
Phase 3 — Summary: Print all results from Phase 3 evaluation.

Reads results/phase3_eval_summary.json and formats for thesis tables.
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    summary_path = Path("results/phase3_eval_summary.json")
    llm_path = Path("results/llm_baseline/metrics.json")

    if not summary_path.exists():
        logger.error("No summary found at %s", summary_path)
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    print("\n" + "="*90)
    print("PHASE 3 RESULTS — FOR THESIS CHAPTER 7 UPDATE")
    print("="*90)

    print("\n--- Main Results (Full Test Set) ---")
    print(f"{'Model':<35} {'F0.5':>8} {'Prec':>8} {'Rec':>8} {'GLEU':>8} {'AgrAcc':>8}")
    print("-" * 80)
    for mid, r in results.items():
        ft = r["full_test"]
        print(f"{mid:<35} {ft['f05']:>8.4f} {ft['precision']:>8.4f} "
              f"{ft['recall']:>8.4f} {ft['gleu']:>8.4f} "
              f"{ft['agreement_accuracy_cer_floored']:>8.4f}")

    print("\n--- Edited Subset Results (source != target) ---")
    print(f"{'Model':<35} {'F0.5':>8} {'Prec':>8} {'Rec':>8} {'GLEU':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 90)
    for mid, r in results.items():
        es = r["edited_subset"]
        print(f"{mid:<35} {es['f05']:>8.4f} {es['precision']:>8.4f} "
              f"{es['recall']:>8.4f} {es['gleu']:>8.4f} "
              f"{es['tp']:>6} {es['fp']:>6} {es['fn']:>6}")

    if llm_path.exists():
        print("\n--- Aya-Expanse-8B Zero-Shot (Edited subset) ---")
        with open(llm_path, "r", encoding="utf-8") as f:
            llm = json.load(f)
        print(f"  F0.5:    {llm['f05']:.4f}")
        print(f"  Prec:    {llm['precision']:.4f}")
        print(f"  Recall:  {llm['recall']:.4f}")
        print(f"  GLEU:    {llm['gleu']:.4f}")
        print(f"  Agr.Acc (CER-floor): {llm['agreement_accuracy_cer_floored']:.4f}")
        print(f"  TP/FP/FN: {llm['tp']}/{llm['fp']}/{llm['fn']}")

    # Augmentation ablation table
    aug_models = {k: v for k, v in results.items() if k.startswith("augment_")}
    no_aug_model = results.get("morphaware_filtered") or results.get("morphaware_v2")
    if aug_models:
        print("\n--- Augmentation Ablation (R34) ---")
        print(f"{'Augment ratio':<20} {'F0.5':>8} {'GLEU':>8}")
        print("-" * 40)
        if no_aug_model:
            es = no_aug_model["edited_subset"]
            print(f"{'0.0 (none)':<20} {es['f05']:>8.4f} {es['gleu']:>8.4f}")
        for mid, r in sorted(aug_models.items()):
            ratio = mid.replace("augment_", "").replace("_", ".")
            es = r["edited_subset"]
            print(f"{ratio:<20} {es['f05']:>8.4f} {es['gleu']:>8.4f}")

    print("\n" + "="*90)


if __name__ == "__main__":
    main()
