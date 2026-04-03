"""Tests for scripts/12_hyperparam_search.py — search space construction and helpers."""

import json
import sys
import os
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import module by path since the filename starts with a number
import importlib.util


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "hyperparam_search",
        Path(__file__).resolve().parent.parent / "scripts" / "12_hyperparam_search.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hp_mod = _load_module()


class TestSearchSpace:
    """Test build_search_space with various configurations."""

    def _make_args(self, **overrides):
        """Create a minimal args namespace."""
        defaults = {
            "model_type": "baseline",
            "lr": None,
            "batch_size_list": None,
            "grad_accum_list": None,
            "agr_weights": None,
        }
        defaults.update(overrides)
        return type("Args", (), defaults)()

    def test_default_baseline_space(self):
        args = self._make_args(model_type="baseline")
        space = hp_mod.build_search_space(args)
        # 4 lr * 3 batch * 2 accum = 24
        assert len(space) == 24
        assert "agreement_loss_weight" not in space[0]

    def test_default_morphaware_space(self):
        args = self._make_args(model_type="morphaware")
        space = hp_mod.build_search_space(args)
        # 4 lr * 3 batch * 2 accum * 3 agr_weight = 72
        assert len(space) == 72
        assert "agreement_loss_weight" in space[0]

    def test_custom_lr_values(self):
        args = self._make_args(lr=[1e-4, 2e-4])
        space = hp_mod.build_search_space(args)
        lrs = {h["lr"] for h in space}
        assert lrs == {1e-4, 2e-4}

    def test_custom_batch_sizes(self):
        args = self._make_args(batch_size_list=[4, 8])
        space = hp_mod.build_search_space(args)
        sizes = {h["batch_size"] for h in space}
        assert sizes == {4, 8}

    def test_custom_grad_accum(self):
        args = self._make_args(grad_accum_list=[2])
        space = hp_mod.build_search_space(args)
        accums = {h["grad_accum_steps"] for h in space}
        assert accums == {2}

    def test_morphaware_custom_agr_weights(self):
        args = self._make_args(model_type="morphaware", agr_weights=[0.0, 1.0])
        space = hp_mod.build_search_space(args)
        weights = {h["agreement_loss_weight"] for h in space}
        assert weights == {0.0, 1.0}

    def test_single_value_per_dim(self):
        args = self._make_args(
            lr=[5e-5], batch_size_list=[16], grad_accum_list=[4]
        )
        space = hp_mod.build_search_space(args)
        assert len(space) == 1
        assert space[0] == {"lr": 5e-5, "batch_size": 16, "grad_accum_steps": 4}

    def test_space_entries_are_dicts(self):
        args = self._make_args()
        space = hp_mod.build_search_space(args)
        for entry in space:
            assert isinstance(entry, dict)
            assert "lr" in entry
            assert "batch_size" in entry
            assert "grad_accum_steps" in entry


class TestLoadConfig:
    """Test YAML config loading."""

    def test_load_valid_config(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("training:\n  batch_size: 32\n", encoding="utf-8")
        cfg = hp_mod.load_config(str(cfg_file))
        assert cfg["training"]["batch_size"] == 32

    def test_load_empty_config(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("", encoding="utf-8")
        cfg = hp_mod.load_config(str(cfg_file))
        assert cfg == {}


class TestLoadPairs:
    """Test JSONL pair loading."""

    def test_load_jsonl(self, tmp_path):
        data = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"source": "src1", "target": "tgt1"}),
            json.dumps({"source": "src2", "target": "tgt2"}),
        ]
        data.write_text("\n".join(lines), encoding="utf-8")
        sources, targets = hp_mod._load_pairs(data)
        assert sources == ["src1", "src2"]
        assert targets == ["tgt1", "tgt2"]

    def test_skip_blank_lines(self, tmp_path):
        data = tmp_path / "blanks.jsonl"
        lines = [
            json.dumps({"source": "a", "target": "b"}),
            "",
            json.dumps({"source": "c", "target": "d"}),
        ]
        data.write_text("\n".join(lines), encoding="utf-8")
        sources, targets = hp_mod._load_pairs(data)
        assert len(sources) == 2
