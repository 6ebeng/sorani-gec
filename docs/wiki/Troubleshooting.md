# Troubleshooting

## Training

**FP16 produces NaN loss.** Known behaviour with ByT5 on this corpus. Train in FP32. The `--fp16` flag exists but every published campaign disabled it; on the RTX 5090 the clean campaign ran FP32 at ~20.7 GiB with:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Morphology-aware model outputs are frozen / identical to input.** Two historical causes, both fixed: (a) gate initialized at 0 makes the morphology pathway an inert identity — use `--morph-gate-init 0.5 --morph-residual-init 0.02`; (b) warmup steps sized for a different dataset scale left the LR near zero for whole epochs — warmup is now computed from the actual step count (~1.5 epochs).

**F₀.₅ stuck near 0.08 with FP ≈ 830.** The category-label contamination signature: check that training data went through `15_clean_corpus.py` and that you are not training on a raw `synthetic_scaled` pool. See [[Training-Campaigns]].

**CUDA OOM.** Drop `--batch-size` and raise `--grad-accum-steps` to keep effective batch 128. At `max_length=512` use batch 8 / accum 16 on a 32 GB card.

## Windows

**Mojibake or `UnicodeEncodeError` when printing Kurdish.**

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**Files opened with wrong encoding.** Every `open()` in this codebase passes `encoding="utf-8"`; do the same in new code. Never rely on the Windows default codepage.

**Paths.** Scripts use `pathlib` relative to the repo root; run them from `Implementation/sorani-gec/`.

## Data & lexicon

**`SoraniLexicon` finds no dictionary.** It searches `data/hunspell/ckb-Arab.dic` then `data/lexicon/ckb-Arab.dic`. Run:

```bash
python scripts/01a_download_ahmadi_lexicon.py
```

**Splits missing.** `data/` is gitignored. Rebuild via the pipeline ([[Data-Pipeline]]) or restore `splits_v2` with `create_splits_v2.py` (dev/test are deterministic given the synthetic corpus and manifest).

**ZWNJ (U+200C) issues.** The normalizer has explicit ZWNJ handling; if a diff view shows "identical" strings that differ, inspect bytes — it is almost always ZWNJ or Arabic vs Kurdish Yeh (ي U+064A vs ی U+06CC).

## Evaluation

**My F₀.₅ does not match the published number.** Check three things: word-level vs span-level scorer (the thesis reports span), `max_length` at generation time (512 for the clean campaign; 256 clips long test sentences), and which campaign/checkpoint you loaded (HF hosts Phase D, not the clean campaign — see [[Reproducibility]]).

**Bootstrap p-values differ slightly between runs.** Resampling is seeded but any change to hypothesis files changes the resample universe; 10,000 resamples give ±0.01 stability.

## Human-evaluation infrastructure

**Do not regenerate the blind set.** `build_eval_pairs.py` overwrites `evaluation_pairs.jsonl` + manifest; the published ratings key against the existing manifest. Only run `analyze_human_eval.py` (read-only) on collected ratings.
