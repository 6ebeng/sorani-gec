"""
Step 9: Export Model to ONNX

Exports a trained BaselineGEC or MorphologyAwareGEC model to ONNX format
for deployment or optimized inference.

For morphology-aware models, exports the encoder with MorphologicalEmbedding,
morph projection, and agreement gating integrated—not just the bare ByT5
encoder (fix 4.6).

Usage:
    python scripts/09_export_onnx.py --checkpoint results/models/baseline/best_model.pt
    python scripts/09_export_onnx.py --checkpoint results/models/morphaware/best_model.pt --morphaware
"""

import argparse
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


class _MorphAwareEncoderWrapper(object):
    """Wrapper that runs the ByT5 encoder + morph integration + agreement gating.

    torch.onnx.export needs a Module whose forward() accepts all inputs and
    returns Tensors.  This wraps the relevant sub-modules of
    MorphologyAwareGEC so the exported graph includes the morphological
    embedding, projection, LayerNorm, agreement bias, and gate.
    """

    def __init__(self, model):
        import torch.nn as nn

        class _Inner(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.encoder = m.backbone.encoder
                self.morph_embedding = m.morph_embedding
                self.morph_projection = m.morph_projection
                self.morph_layer_norm = m.morph_layer_norm
                self.decoder_agr_proj = m.decoder_agr_proj
                self.edge_type_weights = m.edge_type_weights
                self.max_edge_types = m.max_edge_types
                self._build_agreement_bias = m._build_agreement_bias

            def forward(self, input_ids, attention_mask, morph_features, agreement_mask):
                import torch

                enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden = enc.last_hidden_state

                # Morph integration
                morph_emb = self.morph_embedding(morph_features)
                target_len = hidden.size(1)
                if morph_emb.size(1) < target_len:
                    morph_emb = torch.nn.functional.interpolate(
                        morph_emb.transpose(1, 2), size=target_len, mode="nearest",
                    ).transpose(1, 2)
                elif morph_emb.size(1) > target_len:
                    morph_emb = morph_emb[:, :target_len, :]
                combined = torch.cat([hidden, morph_emb], dim=-1)
                hidden = self.morph_layer_norm(self.morph_projection(combined))

                # Agreement gating
                agr_bias = self._build_agreement_bias(agreement_mask, hidden.size(1))
                gate = torch.sigmoid(agr_bias.squeeze(1).max(dim=-1, keepdim=True).values)
                agr_residual = self.decoder_agr_proj(hidden)
                hidden = hidden + gate * agr_residual

                return hidden

        self.inner = _Inner(model)

    def get_module(self):
        return self.inner


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    morphaware: bool = False,
    backbone: str = "google/byt5-small",
    max_length: int = 128,
    opset_version: int = 14,
) -> Path:
    """Export a trained model checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_path: Where to write the .onnx file.
        morphaware: If True, load as MorphologyAwareGEC.
        backbone: Pretrained model name.
        max_length: Max sequence length for dummy input.
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    import torch
    import numpy as np

    # Load the model
    if morphaware:
        from src.model.morphology_aware import MorphologyAwareGEC
        from src.morphology.analyzer import MorphologicalAnalyzer
        analyzer = MorphologicalAnalyzer(use_klpt=False)
        feature_vocab = analyzer.build_feature_vocabulary()
        model = MorphologyAwareGEC(
            model_name=backbone,
            feature_vocab_size=len(feature_vocab),
            max_length=max_length,
        )
    else:
        from src.model.baseline import BaselineGEC
        model = BaselineGEC(model_name=backbone, max_length=max_length)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    # Create dummy input
    tokenizer = model.tokenizer
    dummy_text = "dummy input text"
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="pt",
        max_length=max_length, truncation=True, padding="max_length",
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if morphaware:
        from src.morphology.graph import EDGE_TYPE_ORDER
        num_types = len(EDGE_TYPE_ORDER)
        num_feat = 9  # default morph features
        dummy_morph = torch.zeros(1, max_length, num_feat, dtype=torch.long)
        dummy_agr = torch.zeros(1, num_types, max_length, max_length, dtype=torch.int8)

        wrapper = _MorphAwareEncoderWrapper(model)
        export_module = wrapper.get_module()

        # PyTorch reference output for validation
        with torch.no_grad():
            ref_output = export_module(
                dummy_inputs["input_ids"],
                dummy_inputs["attention_mask"],
                dummy_morph,
                dummy_agr,
            )

        torch.onnx.export(
            export_module,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"],
             dummy_morph, dummy_agr),
            str(out_path),
            input_names=["input_ids", "attention_mask",
                         "morph_features", "agreement_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "morph_features": {0: "batch", 1: "seq"},
                "agreement_mask": {0: "batch"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info("Exported morphology-aware encoder ONNX to %s", out_path)

        # Validation: compare ONNX output vs PyTorch
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(out_path))
            onnx_out = sess.run(None, {
                "input_ids": dummy_inputs["input_ids"].numpy(),
                "attention_mask": dummy_inputs["attention_mask"].numpy(),
                "morph_features": dummy_morph.numpy(),
                "agreement_mask": dummy_agr.numpy().astype(np.float32),
            })
            diff = np.abs(ref_output.numpy() - onnx_out[0]).max()
            logger.info("ONNX validation max diff: %.6e", diff)
            if diff > 1e-4:
                logger.warning("ONNX output diverges from PyTorch (max diff=%.4e)", diff)
        except ImportError:
            logger.info("onnxruntime not installed — skipping ONNX validation")
    else:
        backbone_model = model.model
        encoder = backbone_model.encoder

        torch.onnx.export(
            encoder,
            (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )
        logger.info("Exported baseline encoder ONNX to %s", out_path)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export GEC model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    parser.add_argument("--morphaware", action="store_true")
    parser.add_argument("--backbone", default="google/byt5-small")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--opset", type=int, default=14)
    args = parser.parse_args()

    output = args.output
    if output is None:
        ckpt = Path(args.checkpoint)
        output = str(ckpt.parent / (ckpt.stem + ".onnx"))

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=output,
        morphaware=args.morphaware,
        backbone=args.backbone,
        max_length=args.max_length,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
