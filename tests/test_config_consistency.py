"""
R44 — Config-consistency test.

Verifies that ``configs/default.yaml`` contains the exact hyperparameter
values stated in Chapter 7 of the thesis.  If a researcher edits the YAML
without updating the corresponding prose (or vice versa), this test fails,
making the drift immediately visible in CI.

Expected values are taken directly from the Chapter 7 prose:
  "learning rate 5×10⁻⁵, batch size 16, cosine annealing with 5-epoch
   linear warmup, gradient accumulation over 8 steps, and early stopping
   with patience 5 ... selection_metric val_loss (campaign-2 fix)."
"""

import os
from pathlib import Path

import pytest
import yaml


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_YAML = CONFIGS_DIR / "default.yaml"

# ---------------------------------------------------------------------------
# Expected values — match Chapter 7 §7.1 and §7.3 prose exactly.
# ---------------------------------------------------------------------------
EXPECTED = {
    # training block
    ("training", "learning_rate"):              5.0e-5,
    ("training", "batch_size"):                 16,
    ("training", "weight_decay"):               0.01,
    ("training", "warmup_steps"):               1000,
    ("training", "max_epochs"):                 30,
    ("training", "early_stopping_patience"):    5,
    ("training", "gradient_accumulation_steps"):8,
    ("training", "fp16"):                       False,
    ("training", "selection_metric"):           "val_loss",   # R4 fix
    ("training", "agreement_loss_weight"):      0.3,
    ("training", "gradient_clip_norm"):         1.0,
    # model block
    ("model", "pretrained"):                    "google/byt5-small",
    ("model", "morph_embed_dim"):               64,
}


@pytest.fixture(scope="module")
def cfg() -> dict:
    assert DEFAULT_YAML.exists(), f"Missing config file: {DEFAULT_YAML}"
    with open(DEFAULT_YAML, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.mark.parametrize("keys,expected_value", list(EXPECTED.items()))
def test_config_value(cfg: dict, keys: tuple, expected_value):
    """Each expected hyperparameter must match the YAML value."""
    section, key = keys
    assert section in cfg, f"Section '{section}' missing from default.yaml"
    actual = cfg[section].get(key)
    assert actual is not None, (
        f"Key '{key}' missing from '{section}' in default.yaml"
    )
    assert actual == expected_value, (
        f"Config drift detected!\n"
        f"  {section}.{key}: YAML={actual!r}, expected={expected_value!r}\n"
        f"  Update either configs/default.yaml or the Chapter 7 prose to match."
    )


def test_selection_metric_is_val_loss(cfg: dict):
    """Explicit guard: selection_metric must be 'val_loss' (R4 fix, campaign 2)."""
    actual = cfg.get("training", {}).get("selection_metric")
    assert actual == "val_loss", (
        f"selection_metric is '{actual}', should be 'val_loss'.\n"
        "val_f05 traps training at epoch-1 near-pretrained checkpoints "
        "(F0.5 peaks at epoch 1 because the model copies the source).\n"
        "This is R4 from the reviewer report."
    )


def test_no_fp16(cfg: dict):
    """FP16 must remain disabled — ByT5 byte-level embeddings produce NaN
    loss with fp16 on CUDA 12.x stacks (documented in R21)."""
    assert cfg["training"]["fp16"] is False, (
        "fp16 must remain False: ByT5 NaN loss on CUDA 12.x with fp16."
    )


def test_effective_batch_size(cfg: dict):
    """Effective batch size = batch_size * grad_accum must equal 128."""
    bs = cfg["training"]["batch_size"]
    ga = cfg["training"]["gradient_accumulation_steps"]
    effective = bs * ga
    assert effective == 128, (
        f"Effective batch size is {effective} (batch={bs} × accum={ga}); "
        "should be 128 to match Chapter 7 prose."
    )
