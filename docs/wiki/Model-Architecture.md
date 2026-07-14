# Model Architecture

Three model classes in `src/model/`, all built on ByT5-small.

## Why byte-level (ByT5)

Sorani is written in Arabic script with productive affixation and clitic attachment; subword vocabularies trained on other languages fragment it badly. ByT5 operates on raw UTF-8 bytes, sidestepping tokenization entirely — at the cost of long sequences (one Arabic-script character = 2 bytes).

## `BaselineGEC` (`baseline.py`)

Plain `google/byt5-small` fine-tuned as seq2seq: corrupted sentence in, corrected sentence out. ByT5-small spec: d_model=1472, 12 encoder / 4 decoder layers, 6 heads, ~300M parameters.

## `MorphologyAwareGEC` (`morphology_aware.py`)

Same backbone plus three additions:

1. **`MorphologicalEmbedding`** — 9 separate `nn.Embedding` layers (64-dim each), one per morphological feature, concatenated and projected to 64-dim. Word-level features are aligned to byte positions before injection.
2. **Gated residual injection** — the projected morphology vector enters the encoder representation through a learnable gate. Later campaigns initialize `gate=0.5` and residual weights at 0.02 so morphology contributes from step 1 (a zero-initialized gate trains as an inert identity for many epochs — a real convergence issue found during the campaign-1 audit retrain).
3. **`AgreementPredictor`** — Linear → ReLU → Dropout(0.1) → Linear head over 34 classes (33 agreement-edge types + "correct"), trained with an auxiliary loss.

Combined objective: `L = L_GEC + λ · L_agreement`. The base config uses λ=0.3 with a 5-epoch warmup 0→0.3; the ablation sweep found **λ=0.0 optimal** on dev, so the final campaigns train with the agreement head deconfounded (λ=0).

## `EnsembleGEC` (`ensemble.py`)

Combines baseline + morphology-aware outputs by `majority_vote` or `best_score`. Exploratory only; not part of the released results.

## Training configuration (`configs/default.yaml`)

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 5e-5, cosine schedule with 3 restarts |
| Batch | 16 × grad-accum 8 = effective 128 (clean campaign: 8 × 16 at max_length=512) |
| Precision | **FP32** — FP16 gives NaN loss with ByT5 on this corpus |
| Epochs | 30 max, early stopping patience 5–8 |
| Selection | `val_f05` (validation span F₀.₅) — val-loss selection was tried and abandoned |
| Gradient clip | 1.0 |

CLI flags override YAML; YAML overrides argparse defaults when `--config` is passed.

ONNX export for deployment: `scripts/09_export_onnx.py`.

Next: [[Training-Campaigns]]
