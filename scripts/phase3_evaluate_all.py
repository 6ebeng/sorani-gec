"""
Phase 3 — Evaluate all trained models and produce a summary JSON.

Evaluates:
  - baseline_v2      (R2)
  - morphaware_v2    (R4+R2)
  - morphaware_lambda01  (R15)
  - baseline_filtered    (R1)
  - morphaware_filtered  (R1)
  - augment_0_1 / 0_2 / 0_3  (R34)

On both:
  - Full test set (data/splits/test.jsonl)
  - Edited-only subset (source != target)

Usage:
    python scripts/phase3_evaluate_all.py
"""
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


MODELS = [
    {
        "id": "baseline_p3",
        "path": "results/models/baseline_p3/best_model.pt",
        "morphaware": False,
        "label": "Baseline (R2: val_loss selection, fp16=False)",
    },
    {
        "id": "morphaware_p3",
        "path": "results/models/morphaware_p3/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (R4+R2: FM2+FM1 fix, λ=0.3)",
    },
    {
        "id": "morphaware_lambda01",
        "path": "results/models/morphaware_lambda01/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (FM1+FM2, λ=0.1)",
    },
    {
        "id": "baseline_filtered",
        "path": "results/models/baseline_filtered/best_model.pt",
        "morphaware": False,
        "label": "Baseline (filtered splits, val_loss)",
    },
    {
        "id": "morphaware_filtered",
        "path": "results/models/morphaware_filtered/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (filtered splits, λ=0.1)",
    },
    {
        "id": "augment_0_1",
        "path": "results/models/augment_0_1/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (augment=0.1)",
    },
    {
        "id": "augment_0_2",
        "path": "results/models/augment_0_2/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (augment=0.2)",
    },
    {
        "id": "augment_0_3",
        "path": "results/models/augment_0_3/best_model.pt",
        "morphaware": True,
        "label": "Morphaware (augment=0.3)",
    },
]


def load_test_data(path: Path, edited_only: bool = False):
    sources, references = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src, tgt = rec.get("source", ""), rec.get("target", "")
            if edited_only and src == tgt:
                continue
            sources.append(src)
            references.append(tgt)
    return sources, references


def load_model(model_path: str, morphaware: bool):
    import torch
    from src.model.baseline import BaselineGEC
    from src.model.morphology_aware import MorphologyAwareGEC
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.features import FeatureExtractor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer, feature_extractor = None, None

    if morphaware:
        analyzer = MorphologicalAnalyzer(use_klpt=False)
        feature_vocab = analyzer.build_feature_vocabulary()
        feature_extractor = FeatureExtractor(analyzer=analyzer)
        model = MorphologyAwareGEC(
            model_name="google/byt5-small",
            feature_vocab_size=len(feature_vocab),
            max_length=256,
        )
    else:
        model = BaselineGEC(model_name="google/byt5-small")

    ckpt = Path(model_path)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        logger.info("Loaded: %s", ckpt)
    else:
        logger.warning("Checkpoint not found: %s — using pretrained weights only", ckpt)

    model = model.to(device).eval()
    return model, analyzer, feature_extractor


def evaluate_one(model, sources, references, analyzer=None, feature_extractor=None):
    import torch
    from src.evaluation.f05_scorer import evaluate_corpus
    from src.evaluation.agreement_accuracy import evaluate_agreement_accuracy
    from src.evaluation.gleu_scorer import compute_gleu
    from src.model.morphology_aware import MorphologyAwareGEC

    is_morphaware = isinstance(model, MorphologyAwareGEC) and analyzer is not None
    hypotheses = []
    batch_size = 16

    with torch.no_grad():
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            if is_morphaware:
                hypotheses.extend(model.correct_batch(batch, analyzer, feature_extractor))
            elif hasattr(model, "correct_batch"):
                hypotheses.extend(model.correct_batch(batch))
            else:
                for s in batch:
                    hypotheses.append(model.correct(s))

    f05_m = evaluate_corpus(sources, hypotheses, references)
    agr = evaluate_agreement_accuracy(hypotheses)
    gleu = compute_gleu(sources, hypotheses, references)

    # CER-floored agreement accuracy
    try:
        import editdistance
        avg_cer = sum(
            editdistance.eval(s, h) / max(len(s), 1)
            for s, h in zip(sources, hypotheses)
        ) / max(len(sources), 1)
        cer_floored_agr = 0.0 if avg_cer > 0.5 else agr["accuracy"]
    except ImportError:
        avg_cer = None
        cer_floored_agr = agr["accuracy"]

    return {
        "f05": f05_m.f05,
        "precision": f05_m.precision,
        "recall": f05_m.recall,
        "tp": f05_m.tp,
        "fp": f05_m.fp,
        "fn": f05_m.fn,
        "gleu": gleu,
        "agreement_accuracy": agr["accuracy"],
        "agreement_accuracy_cer_floored": cer_floored_agr,
        "avg_cer": avg_cer,
        "n_pairs": len(sources),
    }


def main():
    test_full_path = Path("data/splits/test.jsonl")
    test_filt_path = Path("data/splits_filtered/test.jsonl")
    output_path = Path("results/phase3_eval_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sources_full, refs_full = load_test_data(test_full_path, edited_only=False)
    sources_edited, refs_edited = load_test_data(test_full_path, edited_only=True)
    logger.info("Full test: %d pairs; Edited subset: %d pairs",
                len(sources_full), len(sources_edited))

    sources_filt_full, refs_filt_full = [], []
    sources_filt_edited, refs_filt_edited = [], []
    if test_filt_path.exists():
        sources_filt_full, refs_filt_full = load_test_data(test_filt_path, edited_only=False)
        sources_filt_edited, refs_filt_edited = load_test_data(test_filt_path, edited_only=True)
        logger.info("Filtered full: %d; filtered edited: %d",
                    len(sources_filt_full), len(sources_filt_edited))

    all_results = {}

    for model_cfg in MODELS:
        mid = model_cfg["id"]
        logger.info("--- Evaluating: %s ---", mid)

        if not Path(model_cfg["path"]).exists():
            logger.warning("Checkpoint missing for %s — skipping", mid)
            continue

        model, analyzer, feature_extractor = load_model(
            model_cfg["path"], model_cfg["morphaware"]
        )

        # Choose correct test set
        is_filtered = "filtered" in mid
        if is_filtered:
            s_full, r_full = sources_filt_full, refs_filt_full
            s_edit, r_edit = sources_filt_edited, refs_filt_edited
        else:
            s_full, r_full = sources_full, refs_full
            s_edit, r_edit = sources_edited, refs_edited

        result_full = evaluate_one(model, s_full, r_full, analyzer, feature_extractor)
        result_edit = evaluate_one(model, s_edit, r_edit, analyzer, feature_extractor)

        all_results[mid] = {
            "label": model_cfg["label"],
            "morphaware": model_cfg["morphaware"],
            "checkpoint": model_cfg["path"],
            "full_test": result_full,
            "edited_subset": result_edit,
        }

        # Save per-model JSON
        per_model_path = Path(model_cfg["path"]).parent / "phase3_metrics.json"
        with open(per_model_path, "w", encoding="utf-8") as f:
            json.dump(all_results[mid], f, indent=2, ensure_ascii=False)

        logger.info(
            "%s | full F0.5=%.4f | edited F0.5=%.4f",
            mid, result_full["f05"], result_edit["f05"]
        )

        # Free GPU memory before next model
        import torch
        del model
        torch.cuda.empty_cache()

    # Save combined summary
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info("Summary saved to %s", output_path)
    print("\n=== Phase 3 Evaluation Summary ===")
    print(f"{'Model':<30} {'Full F0.5':>10} {'Edit F0.5':>10} {'Edit GLEU':>10}")
    print("-" * 65)
    for mid, r in all_results.items():
        print(f"{mid:<30} {r['full_test']['f05']:>10.4f} "
              f"{r['edited_subset']['f05']:>10.4f} "
              f"{r['edited_subset']['gleu']:>10.4f}")


if __name__ == "__main__":
    main()
