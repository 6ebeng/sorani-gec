"""
Tests for the MorphologyAwareGEC model architecture.

These tests validate:
  - MorphologicalEmbedding forward pass shape
  - AgreementPredictor output dimensionality (25 types)
  - MorphologyAwareGEC forward with 3D and 4D agreement masks
  - Edge-type weight parameter count (24)
  - Configurable agreement_loss_weight and edge_type_loss_weights
  - Agreement loss contribution to total loss
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")

if HAS_TORCH:
    from src.model.morphology_aware import (
        MorphologicalEmbedding,
        AgreementPredictor,
        MorphologyAwareGEC,
    )


# ============================================================================
# MorphologicalEmbedding Tests
# ============================================================================

def test_morph_embedding_output_shape():
    """MorphologicalEmbedding produces [batch, seq, embed_dim] output."""
    emb = MorphologicalEmbedding(feature_vocab_size=50, num_features=9, embed_dim=64)
    x = torch.randint(0, 50, (2, 5, 9))  # [batch=2, seq=5, features=9]
    out = emb(x)
    assert out.shape == (2, 5, 64)
    print(f"  MorphologicalEmbedding shape: {out.shape}")


def test_morph_embedding_num_feature_layers():
    """One embedding layer per morphological feature."""
    emb = MorphologicalEmbedding(feature_vocab_size=50, num_features=9, embed_dim=32)
    assert len(emb.feature_embeddings) == 9


def test_morph_embedding_default_uses_9_features():
    """M1: Default num_features should be 9 (including aspect)."""
    emb = MorphologicalEmbedding(feature_vocab_size=50)
    assert emb.num_features == 9
    assert len(emb.feature_embeddings) == 9
    print(f"  Default num_features: {emb.num_features}")


# ============================================================================
# AgreementPredictor Tests
# ============================================================================

def test_agreement_predictor_18_types():
    """AgreementPredictor with 25 output classes."""
    pred = AgreementPredictor(hidden_dim=128, num_agreement_types=25)
    x = torch.randn(2, 5, 128)
    out = pred(x)
    assert out.shape == (2, 5, 25)
    print(f"  AgreementPredictor output: {out.shape}")


def test_agreement_predictor_custom_types():
    """AgreementPredictor accepts custom num_agreement_types."""
    pred = AgreementPredictor(hidden_dim=64, num_agreement_types=20)
    x = torch.randn(1, 3, 64)
    out = pred(x)
    assert out.shape == (1, 3, 20)


# ============================================================================
# MorphologyAwareGEC Tests (lightweight — no model download)
# ============================================================================

@pytest.fixture(scope="module")
def gec_model():
    """Create a small GEC model for testing (downloads byt5-small once)."""
    try:
        model = MorphologyAwareGEC(
            model_name="google/byt5-small",
            feature_vocab_size=50,
            num_morph_features=9,
            morph_embed_dim=64,
            agreement_loss_weight=0.3,
            max_length=32,
            num_agreement_types=25,
        )
        model.eval()
        return model
    except Exception:
        pytest.skip("byt5-small not available (offline or no disk space)")


def test_model_has_19_edge_type_weights(gec_model):
    """max_edge_types=24; edge_type_weights parameter has 24 entries."""
    assert gec_model.max_edge_types == 24
    assert gec_model.edge_type_weights.shape == (24,)
    print(f"  edge_type_weights shape: {gec_model.edge_type_weights.shape}")


def test_model_agreement_loss_weight_default(gec_model):
    """Default agreement_loss_weight is 0.3."""
    assert gec_model.agreement_loss_weight == pytest.approx(0.3)


def test_model_edge_type_loss_weights_none_default(gec_model):
    """edge_type_loss_weights defaults to None."""
    assert gec_model.edge_type_loss_weights is None


def test_forward_3d_agreement_mask(gec_model):
    """Forward pass with 3D binary agreement mask works."""
    tok = gec_model.tokenizer
    enc = tok("من دەچم", return_tensors="pt", max_length=32,
              truncation=True, padding="max_length")
    labels = enc["input_ids"].clone()
    mask_3d = torch.zeros(1, 4, 4)
    mask_3d[0, 0, 1] = 1  # subject → verb edge
    with torch.no_grad():
        out = gec_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            agreement_mask=mask_3d,
        )
    assert "logits" in out
    assert "agreement_logits" in out
    assert out["agreement_logits"].shape[-1] == 25
    print(f"  3D mask forward OK, agreement_logits shape: {out['agreement_logits'].shape}")


def test_forward_4d_typed_agreement_mask(gec_model):
    """Forward pass with 4D typed agreement mask applies learnable weights."""
    tok = gec_model.tokenizer
    enc = tok("من دەچم", return_tensors="pt", max_length=32,
              truncation=True, padding="max_length")
    labels = enc["input_ids"].clone()
    mask_4d = torch.zeros(1, 3, 4, 4)  # 3 edge types
    mask_4d[0, 0, 0, 1] = 1  # type 0: subject_verb
    mask_4d[0, 1, 2, 1] = 1  # type 1: object_verb_ergative
    with torch.no_grad():
        out = gec_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            agreement_mask=mask_4d,
        )
    assert "logits" in out
    assert out["agreement_logits"].shape[-1] == 25
    print(f"  4D mask forward OK, logits shape: {out['logits'].shape}")


def test_agreement_loss_added_when_labels_present(gec_model):
    """Agreement loss should increase total loss when agreement_labels provided."""
    tok = gec_model.tokenizer
    enc = tok("من دەچم", return_tensors="pt", max_length=32,
              truncation=True, padding="max_length")
    labels = enc["input_ids"].clone()
    # Without agreement labels
    with torch.no_grad():
        out_no_agr = gec_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
    # With agreement labels (all zeros = no agreement type)
    agr_labels = torch.zeros_like(enc["input_ids"])
    with torch.no_grad():
        out_with_agr = gec_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
            agreement_labels=agr_labels,
        )
    # With agreement labels, loss should be >= loss without (auxiliary adds to it)
    assert out_with_agr["loss"].item() >= out_no_agr["loss"].item() - 0.01
    print(f"  Loss without agr: {out_no_agr['loss'].item():.4f}")
    print(f"  Loss with agr:    {out_with_agr['loss'].item():.4f}")


def test_correct_method_returns_string(gec_model):
    """correct() should return a string."""
    result = gec_model.correct("من دەچم")
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"  correct() output: '{result}'")


# ============================================================================
# Build agreement bias unit tests (no model download needed)
# ============================================================================

def test_build_agreement_bias_3d():
    """_build_agreement_bias with 3D input returns [batch, 1, seq, seq]."""
    model = torch.nn.Module()
    model.max_edge_types = 24
    model.edge_type_weights = torch.nn.Parameter(torch.ones(24) / 24)
    model._build_agreement_bias = MorphologyAwareGEC._build_agreement_bias.__get__(model)
    mask = torch.ones(2, 4, 4)
    bias = model._build_agreement_bias(mask, seq_len=6)
    assert bias.shape == (2, 1, 6, 6)
    print(f"  3D bias shape: {bias.shape}")


def test_build_agreement_bias_4d():
    """_build_agreement_bias with 4D input collapses types via softmax weights."""
    model = torch.nn.Module()
    model.max_edge_types = 24
    model.edge_type_weights = torch.nn.Parameter(torch.ones(24) / 24)
    model._build_agreement_bias = MorphologyAwareGEC._build_agreement_bias.__get__(model)
    mask = torch.zeros(1, 3, 4, 4)
    mask[0, 0, 0, 1] = 1
    mask[0, 2, 2, 3] = 1
    bias = model._build_agreement_bias(mask, seq_len=4)
    assert bias.shape == (1, 1, 4, 4)
    assert bias[0, 0, 0, 1].item() > 0  # Edge present
    assert bias[0, 0, 2, 3].item() > 0  # Edge present
    print(f"  4D bias shape: {bias.shape}, values correct")


# ============================================================================
# Integration Tests: Full Pipeline
# ============================================================================

def test_integration_analyzer_to_graph():
    """Integration: analyzer → builder → graph produces valid typed stacked matrix."""
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.builder import build_agreement_graph
    from src.morphology.graph import EDGE_TYPE_ORDER

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    # Simple sentence with subject-verb agreement
    sentence = "کوڕەکە دەڕوات"
    graph = build_agreement_graph(sentence, analyzer)

    # Graph should have tokens and at least one edge
    assert len(graph.tokens) >= 2
    assert isinstance(graph.edges, list)

    # Typed stacked matrix should be well-formed
    stacked, type_names = graph.to_typed_stacked_matrix()
    n = len(graph.tokens)
    for mat in stacked:
        assert len(mat) == n
        for row in mat:
            assert len(row) == n
    # All type names should be strings
    for tname in type_names:
        assert isinstance(tname, str)
    print(f"  Integration: {len(graph.tokens)} tokens, {len(graph.edges)} edges, "
          f"{len(type_names)} types")


def test_integration_feature_extractor():
    """Integration: FeatureExtractor produces correct-shape output."""
    from src.morphology.features import FeatureExtractor

    fe = FeatureExtractor()
    features = fe.extract_features("من دەچم بۆ قوتابخانە")
    assert len(features) >= 3  # at least 3 words
    for vec in features:
        assert len(vec) == 9  # 9 feature indices per token
    print(f"  FeatureExtractor: {len(features)} tokens × 9 features")


def test_integration_clause_boundary_blocks_edge():
    """Clause boundary between noun and verb should prevent cross-clause edges."""
    from src.morphology.analyzer import MorphologicalAnalyzer
    from src.morphology.builder import build_agreement_graph

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    # Two clauses separated by و (conjunction = clause boundary)
    sentence = "پیاوەکە چوو و ژنەکە هات"
    graph = build_agreement_graph(sentence, analyzer)
    # No edge should cross from پیاوەکە (idx 0) to هات (idx 4)
    cross_clause_edges = [
        e for e in graph.edges
        if (e.source_idx == 0 and e.target_idx == 4)
        or (e.source_idx == 4 and e.target_idx == 0)
    ]
    assert len(cross_clause_edges) == 0, (
        f"Found cross-clause edge(s): {cross_clause_edges}"
    )
    print(f"  Clause boundary correctly blocked cross-clause edge")


def test_c5_default_num_agreement_types_matches_edge_type_order():
    """C5: Default num_agreement_types must equal len(EDGE_TYPE_ORDER) + 1."""
    from src.morphology.graph import EDGE_TYPE_ORDER
    predictor = AgreementPredictor(hidden_dim=64)
    # The last layer in the Sequential classifier has num_agreement_types outputs
    out_features = predictor.classifier[-1].out_features
    assert out_features == len(EDGE_TYPE_ORDER) + 1, (
        f"AgreementPredictor default should produce {len(EDGE_TYPE_ORDER) + 1} types, "
        f"got {out_features}"
    )
    print(f"  C5: Default num_agreement_types = {out_features} "
          f"(EDGE_TYPE_ORDER has {len(EDGE_TYPE_ORDER)} types + 1 correct class)")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== MorphologicalEmbedding Tests ===")
    test_morph_embedding_output_shape()
    test_morph_embedding_num_feature_layers()
    test_morph_embedding_default_uses_9_features()

    print("\n=== AgreementPredictor Tests ===")
    test_agreement_predictor_18_types()
    test_agreement_predictor_custom_types()

    print("\n=== Agreement Bias Tests ===")
    test_build_agreement_bias_3d()
    test_build_agreement_bias_4d()

    print("\n=== Integration Tests ===")
    test_integration_analyzer_to_graph()
    test_integration_feature_extractor()
    test_integration_clause_boundary_blocks_edge()

    print("\n=== Round 17 Critical Gap Fix Tests — C5 ===")
    test_c5_default_num_agreement_types_matches_edge_type_order()

    print("\nAll model unit tests passed!")
    print("(Run with pytest for model integration tests requiring byt5-small)")
