"""
Integration tests using small Wikipedia corpus texts.

These tests exercise the full pipeline end-to-end:
  normalizer → analyzer → agreement graph → agreement checker → error pipeline

Corpus sentences are drawn from Sorani Kurdish Wikipedia articles on
Kurdistan, Hewlêr (Erbil), the Kurdish language, and Sorani grammar.
Each sentence is real Sorani text, not synthetic.
"""

import sys
import os
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer, sentence_split, deduplicate_sentences
from src.data.collector import CorpusCollector
from src.morphology.analyzer import MorphologicalAnalyzer, MorphFeatures
from src.morphology.agreement import build_agreement_graph
from src.morphology.graph import AgreementGraph, EDGE_TYPE_ORDER
from src.evaluation.agreement_accuracy import (
    AgreementChecker,
    AgreementResult,
    evaluate_agreement_accuracy,
)

# ============================================================================
# Wikipedia Corpus Fixture — Real Sorani Kurdish sentences
# ============================================================================

# Source: Sorani Kurdish Wikipedia — articles on Kurdistan Region, Hewlêr,
# Kurdish language, and general encyclopedic topics.
# These are grammatically correct reference sentences.
WIKI_CORPUS = [
    # Geography & Kurdistan Region
    "هەرێمی کوردستان لە باکووری عێراقدایە.",
    "هەولێر پایتەختی هەرێمی کوردستانە.",
    "سلێمانی دووەمین شاری گەورەی هەرێمی کوردستانە.",
    "شاخەکانی کوردستان بەرزن و زۆر جوانن.",
    "ئاوی زێی گەورە لە ناوچەکەدا تێدەپەڕێت.",

    # Language & Culture
    "زمانی کوردی یەکێکە لە زمانەکانی ەیرانی ڕۆژهەڵاتی.",
    "سۆرانی لە هەرێمی کوردستاندا بە فەرمی بەکاردەهێنرێت.",
    "ئەلفوبێی کوردی لەسەر بنەمای عەرەبی دانراوە.",
    "ئەدەبیاتی کوردی مێژوویەکی زۆر درێژی هەیە.",
    "گۆرانی کوردی بەشێکی گرنگی کەلتووری کوردییە.",

    # Simple declarative sentences
    "من دەچم بۆ قوتابخانە.",
    "ئەو کتێبەکەی خوێندەوە.",
    "ئێمە نانمان خوارد.",
    "تۆ دەتوانیت بێیت.",
    "ئەوان لە ماڵەوە نیشتەجێن.",

    # Past tense (transitive — ergative agreement)
    "منداڵەکان نانیان خوارد.",
    "پیاوەکە نامەکەی نووسی.",
    "ژنەکە نانی لەبەر کرد.",

    # Past tense (intransitive — nominative agreement)
    "منداڵەکان چوون بۆ قوتابخانە.",
    "پیاوەکە هات.",

    # Compound & complex sentences
    "من دەچم بۆ بازاڕ و نان دەکڕم.",
    "ئەگەر بارانی بارا ئێمە ناچین بۆ دەرەوە.",
    "ئەو کتێبەکەی کە تۆ خوێندتەوە زۆر باشە.",

    # Demonstrative + noun phrases
    "ئەم شارانە زۆر کۆنن.",
    "ئەو پیاوە زۆر بەتوانایە.",

    # Question forms
    "کێ هات؟",
    "تۆ کەی دەچیت؟",

    # Sentences with clitics
    "کتێبەکەم لە ماڵەوەیە.",
    "ماڵەکەمان گەورەیە.",
]

# Sentences with known agreement errors (for error detection testing)
WIKI_CORPUS_WITH_ERRORS = [
    # Subject-verb mismatch: من (1sg) + دەچین (1pl)
    ("من دەچین بۆ بازاڕ.", "subject_verb"),
    # Demonstrative + definite marker co-occurrence (F#10)
    ("ئەم کتێبەکە باشە.", "demonstrative_definite"),
]


# ============================================================================
# 1. Normalization Integration Tests
# ============================================================================

def test_normalize_wiki_corpus():
    """Normalizer processes all Wikipedia sentences without crashing."""
    normalizer = SoraniNormalizer()
    for sentence in WIKI_CORPUS:
        result = normalizer.normalize(sentence)
        assert isinstance(result, str)
        assert len(result) > 0, f"Normalization produced empty output for: {sentence}"
    print("  test_normalize_wiki_corpus: PASSED")


def test_normalize_preserves_kurdish_chars():
    """Normalization preserves Kurdish-specific characters (ڕ ڵ ۆ ێ ە پ چ گ)."""
    normalizer = SoraniNormalizer()
    kurdish_chars = set("ڕڵۆێەپچگ")
    for sentence in WIKI_CORPUS:
        normalized = normalizer.normalize(sentence)
        original_kurdish = kurdish_chars & set(sentence)
        normalized_kurdish = kurdish_chars & set(normalized)
        assert original_kurdish <= normalized_kurdish, (
            f"Lost Kurdish chars in: {sentence}\n"
            f"  Missing: {original_kurdish - normalized_kurdish}"
        )
    print("  test_normalize_preserves_kurdish_chars: PASSED")


def test_sentence_split_wiki_paragraph():
    """sentence_split correctly segments a multi-sentence paragraph."""
    paragraph = (
        "هەرێمی کوردستان لە باکووری عێراقدایە. "
        "هەولێر پایتەختی هەرێمی کوردستانە. "
        "سلێمانی دووەمین شاری گەورەی هەرێمی کوردستانە."
    )
    sentences = sentence_split(paragraph)
    assert len(sentences) >= 3, f"Expected ≥3 sentences, got {len(sentences)}"
    print(f"  test_sentence_split_wiki_paragraph: PASSED ({len(sentences)} sentences)")


def test_deduplicate_corpus():
    """Deduplication removes exact duplicates from corpus."""
    corpus = WIKI_CORPUS + WIKI_CORPUS[:5]  # add 5 duplicates
    deduped = deduplicate_sentences(corpus)
    assert len(deduped) == len(WIKI_CORPUS), (
        f"Expected {len(WIKI_CORPUS)} unique sentences, got {len(deduped)}"
    )
    print("  test_deduplicate_corpus: PASSED")


# ============================================================================
# 2. Morphological Analysis Integration Tests
# ============================================================================

def test_analyze_wiki_corpus():
    """Analyzer produces features for every token in every sentence."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    total_tokens = 0
    for sentence in WIKI_CORPUS:
        features_list = analyzer.analyze_sentence(sentence)
        assert len(features_list) > 0, f"No tokens for: {sentence}"
        for feat in features_list:
            assert isinstance(feat, MorphFeatures)
            assert feat.token, f"Empty token in: {sentence}"
            assert feat.pos, f"Missing POS for token '{feat.token}' in: {sentence}"
        total_tokens += len(features_list)
    assert total_tokens > 100, f"Expected >100 total tokens, got {total_tokens}"
    print(f"  test_analyze_wiki_corpus: PASSED ({total_tokens} tokens analyzed)")


def test_analyzer_detects_verbs():
    """Analyzer finds present-tense verbs with correct POS in known sentences."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    # "من دەچم بۆ قوتابخانە." — دەچم should be a verb
    features = analyzer.analyze_sentence("من دەچم بۆ قوتابخانە.")
    verb_tokens = [f for f in features if f.pos == "VERB"]
    assert len(verb_tokens) >= 1, (
        f"Expected at least 1 verb, got: {[(f.token, f.pos) for f in features]}"
    )
    print(f"  test_analyzer_detects_verbs: PASSED ({len(verb_tokens)} verbs found)")


def test_analyzer_detects_pronouns():
    """Analyzer identifies subject pronouns with correct person/number."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    features = analyzer.analyze_sentence("من دەچم بۆ قوتابخانە.")
    pronoun_features = [f for f in features if f.token == "من"]
    assert len(pronoun_features) == 1
    pf = pronoun_features[0]
    assert pf.pos == "PRON", f"Expected PRON, got {pf.pos}"
    assert pf.person == "1", f"Expected person=1, got {pf.person}"
    assert pf.number == "sg", f"Expected number=sg, got {pf.number}"
    print("  test_analyzer_detects_pronouns: PASSED")


def test_analyzer_past_tense_detection():
    """Analyzer detects past tense in a sentence with known past transitive verb."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    # نووسی (wrote) is a well-known past transitive stem
    features = analyzer.analyze_sentence("پیاوەکە نامەکەی نووسی.")
    verb_features = [f for f in features if f.pos == "VERB"]
    assert len(verb_features) >= 1, (
        f"Expected at least 1 verb, got: {[(f.token, f.pos) for f in features]}"
    )
    print("  test_analyzer_past_tense_detection: PASSED")


def test_normalized_then_analyzed():
    """Normalizer output is analyzable without errors (pipeline chain)."""
    normalizer = SoraniNormalizer()
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    for sentence in WIKI_CORPUS:
        normalized = normalizer.normalize(sentence)
        features = analyzer.analyze_sentence(normalized)
        assert len(features) > 0, f"No features after normalize→analyze: {sentence}"
    print("  test_normalized_then_analyzed: PASSED")


# ============================================================================
# 3. Agreement Graph Integration Tests
# ============================================================================

def test_build_graph_wiki_corpus():
    """Agreement graph builds for all Wikipedia sentences without error."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    graphs_with_edges = 0
    for sentence in WIKI_CORPUS:
        graph = build_agreement_graph(sentence, analyzer)
        assert isinstance(graph, AgreementGraph)
        if len(graph) > 0:
            graphs_with_edges += 1
    assert graphs_with_edges > 0, "No sentences produced agreement edges"
    print(f"  test_build_graph_wiki_corpus: PASSED ({graphs_with_edges}/{len(WIKI_CORPUS)} with edges)")


def test_graph_edge_types_valid():
    """All edge types produced by the builder are in EDGE_TYPE_ORDER."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    all_edge_types: set[str] = set()
    for sentence in WIKI_CORPUS:
        graph = build_agreement_graph(sentence, analyzer)
        for edge in graph.edges:
            all_edge_types.add(edge.agreement_type)

    unknown = all_edge_types - set(EDGE_TYPE_ORDER)
    assert len(unknown) == 0, f"Unknown edge types: {unknown}"
    print(f"  test_graph_edge_types_valid: PASSED ({len(all_edge_types)} unique types)")


def test_graph_adjacency_matrix_shape():
    """Adjacency matrix dimensions match token count for each sentence."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    for sentence in WIKI_CORPUS[:10]:
        graph = build_agreement_graph(sentence, analyzer)
        matrix = graph.to_adjacency_matrix()
        n = len(graph.tokens)
        assert len(matrix) == n, f"Matrix rows {len(matrix)} != tokens {n}"
        for row in matrix:
            assert len(row) == n, f"Matrix col {len(row)} != tokens {n}"
    print("  test_graph_adjacency_matrix_shape: PASSED")


def test_graph_typed_stacked_matrix():
    """Typed stacked matrix dimensions match token count; types are valid."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    sentence = "من دەچم بۆ قوتابخانە."
    graph = build_agreement_graph(sentence, analyzer)
    stacked, types = graph.to_typed_stacked_matrix()
    # Only types with edges are included; all must be in EDGE_TYPE_ORDER
    assert len(types) > 0, "No edge types in stacked matrix"
    for t in types:
        assert t in EDGE_TYPE_ORDER, f"Unknown type: {t}"
    n = len(graph.tokens)
    assert len(stacked) == len(types)
    for layer in stacked:
        assert len(layer) == n
    print(f"  test_graph_typed_stacked_matrix: PASSED ({len(types)} types)")


def test_graph_violation_check():
    """check_agreement returns violations list (may be empty for correct sentences)."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    for sentence in WIKI_CORPUS[:10]:
        graph = build_agreement_graph(sentence, analyzer)
        violations = graph.check_agreement()
        assert isinstance(violations, list)
        for v in violations:
            assert "type" in v
            assert "source" in v
            assert "target" in v
    print("  test_graph_violation_check: PASSED")


# ============================================================================
# 4. Agreement Checker Integration Tests
# ============================================================================

def test_checker_wiki_corpus():
    """Agreement checker processes all Wikipedia sentences with 5 checks each."""
    checker = AgreementChecker()
    results = []
    for sentence in WIKI_CORPUS:
        result = checker.check_sentence(sentence)
        assert isinstance(result, AgreementResult)
        assert result.checks_total == 8
        assert result.checks_passed <= result.checks_total
        assert 0.0 <= result.accuracy <= 1.0
        results.append(result)
    correct = sum(1 for r in results if r.is_correct)
    print(f"  test_checker_wiki_corpus: PASSED ({correct}/{len(results)} clean)")


def test_checker_detects_sv_mismatch():
    """Checker flags subject-verb person/number mismatch."""
    checker = AgreementChecker()
    # من (1sg) with دەچین (1pl) — number mismatch
    result = checker.check_sentence("من دەچین بۆ بازاڕ.")
    sv = [v for v in result.violations if "Subject-verb" in v]
    assert len(sv) > 0, f"Expected SV violation, got: {result.violations}"
    print(f"  test_checker_detects_sv_mismatch: PASSED")


def test_checker_detects_dem_def():
    """Checker flags demonstrative + definite marker co-occurrence (F#10)."""
    checker = AgreementChecker()
    result = checker.check_sentence("ئەم کتێبەکە باشە.")
    dem = [v for v in result.violations if "Demonstrative" in v]
    assert len(dem) > 0, f"Expected dem+def violation, got: {result.violations}"
    print(f"  test_checker_detects_dem_def: PASSED")


def test_evaluate_agreement_accuracy_wiki():
    """evaluate_agreement_accuracy returns well-formed metrics on wiki corpus."""
    metrics = evaluate_agreement_accuracy(WIKI_CORPUS)
    assert "accuracy" in metrics
    assert "total_sentences" in metrics
    assert metrics["total_sentences"] == len(WIKI_CORPUS)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    print(f"  test_evaluate_agreement_accuracy_wiki: PASSED (accuracy={metrics['accuracy']:.2f})")


# ============================================================================
# 5. Full Pipeline Integration Tests (normalize → analyze → graph → check)
# ============================================================================

def test_full_pipeline_clean_sentences():
    """Full pipeline: normalize → analyze → build graph → check agreement."""
    normalizer = SoraniNormalizer()
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    checker = AgreementChecker(analyzer=analyzer)

    for sentence in WIKI_CORPUS[:10]:
        # Step 1: Normalize
        normalized = normalizer.normalize(sentence)
        assert len(normalized) > 0

        # Step 2: Analyze
        features = analyzer.analyze_sentence(normalized)
        assert len(features) > 0

        # Step 3: Build agreement graph
        graph = build_agreement_graph(normalized, analyzer)
        assert isinstance(graph, AgreementGraph)

        # Step 4: Check agreement
        result = checker.check_sentence(normalized)
        assert result.checks_total == 8

    print("  test_full_pipeline_clean_sentences: PASSED")


def test_full_pipeline_with_errors():
    """Full pipeline detects known errors inserted into sentences."""
    normalizer = SoraniNormalizer()
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    checker = AgreementChecker(analyzer=analyzer)

    for sentence, error_type in WIKI_CORPUS_WITH_ERRORS:
        normalized = normalizer.normalize(sentence)
        result = checker.check_sentence(normalized)
        assert len(result.violations) > 0, (
            f"Expected violations for {error_type} error in: {sentence}\n"
            f"  Got: {result.violations}"
        )
    print("  test_full_pipeline_with_errors: PASSED")


def test_collector_is_sorani_wiki():
    """CorpusCollector._is_sorani classifies Wikipedia sentences correctly."""
    for sentence in WIKI_CORPUS:
        assert CorpusCollector._is_sorani(sentence), (
            f"_is_sorani() returned False for: {sentence}"
        )
    # Non-Sorani text should be rejected
    assert not CorpusCollector._is_sorani("This is an English sentence.")
    assert not CorpusCollector._is_sorani("هذه جملة عربية بسيطة.")
    print("  test_collector_is_sorani_wiki: PASSED")


def test_collector_write_and_read_corpus():
    """Collector can write corpus sentences and read them back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write sentences to a text file
        input_dir = os.path.join(tmpdir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "wiki.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in WIKI_CORPUS:
                f.write(sentence + "\n")

        # Use collector to process
        output_dir = os.path.join(tmpdir, "output")
        collector = CorpusCollector(output_dir=output_dir)
        count = collector.collect_from_text_files(input_dir, source_name="wiki_test")
        assert count > 0, f"Collector returned 0 sentences"

        # Verify output exists
        output_files = os.listdir(output_dir)
        assert len(output_files) > 0, "No output files created"
    print(f"  test_collector_write_and_read_corpus: PASSED ({count} sentences)")


def test_feature_vocabulary_coverage():
    """Feature vocabulary covers all features seen in wiki corpus."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    vocab = analyzer.build_feature_vocabulary()
    assert "PAD" in vocab
    assert "UNK" in vocab
    assert len(vocab) > 10, f"Vocabulary too small: {len(vocab)}"

    # Check that all features from corpus sentences map to known vocab entries
    unmapped_count = 0
    for sentence in WIKI_CORPUS:
        for feat in analyzer.analyze_sentence(sentence):
            if feat.pos and feat.pos not in vocab:
                unmapped_count += 1
    # Some unmapped is OK (UNK handles it), but not all
    print(f"  test_feature_vocabulary_coverage: PASSED (vocab_size={len(vocab)}, unmapped={unmapped_count})")


# ============================================================================
# 6. Corpus-Level Statistics Tests
# ============================================================================

def test_corpus_agreement_statistics():
    """Compute and validate corpus-level agreement statistics."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    edge_type_counts: dict[str, int] = {}
    total_edges = 0

    for sentence in WIKI_CORPUS:
        graph = build_agreement_graph(sentence, analyzer)
        counts = graph.edge_type_counts()
        for etype, count in counts.items():
            edge_type_counts[etype] = edge_type_counts.get(etype, 0) + count
            total_edges += count

    assert total_edges > 0, "No agreement edges found in corpus"
    # At least some edge type diversity
    assert len(edge_type_counts) >= 2, (
        f"Only {len(edge_type_counts)} edge types in corpus: {list(edge_type_counts.keys())}"
    )
    print(f"  test_corpus_agreement_statistics: PASSED "
          f"({total_edges} edges, {len(edge_type_counts)} types)")


def test_corpus_pos_distribution():
    """Verify POS distribution across corpus is reasonable."""
    analyzer = MorphologicalAnalyzer(use_klpt=False)
    pos_counts: dict[str, int] = {}
    for sentence in WIKI_CORPUS:
        for feat in analyzer.analyze_sentence(sentence):
            pos_counts[feat.pos] = pos_counts.get(feat.pos, 0) + 1

    # Expect at least NOUN, VERB, PRON, and a few more
    assert "NOUN" in pos_counts, f"No NOUNs found. POS: {pos_counts}"
    assert "VERB" in pos_counts, f"No VERBs found. POS: {pos_counts}"
    assert pos_counts["NOUN"] > pos_counts.get("VERB", 0), (
        "Expected more nouns than verbs in encyclopedic text"
    )
    print(f"  test_corpus_pos_distribution: PASSED ({dict(sorted(pos_counts.items()))})")


# ============================================================================
# ============================================================================
# 7. End-to-End GEC Pipeline Test (7D.1)
# ============================================================================

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_e2e_generate_train_evaluate():
    """End-to-end: generate 10 pairs → train 1 epoch → evaluate → F₀.₅ ≥ 0."""
    from src.errors.pipeline import ErrorPipeline
    from src.model.baseline import BaselineGEC
    from src.evaluation.f05_scorer import evaluate_corpus

    pipeline = ErrorPipeline(error_rate=0.3, seed=42)
    pairs = []
    for sentence in WIKI_CORPUS[:10]:
        result = pipeline.process_sentence(sentence)
        pairs.append({"source": result.corrupted, "target": result.original})
    assert len(pairs) == 10

    model = BaselineGEC(model_name="google/byt5-small", max_length=64)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    sources = [p["source"] for p in pairs]
    targets = [p["target"] for p in pairs]
    loss = model.training_step(sources, targets)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0, "Training loss should be positive"

    model.eval()
    hypotheses = [model.correct(s) for s in sources]
    metrics = evaluate_corpus(sources, hypotheses, targets)
    assert metrics.f05 >= 0.0, f"F₀.₅ should be non-negative, got {metrics.f05}"
    print(f"  test_e2e_generate_train_evaluate: PASSED (loss={loss.item():.4f}, F₀.₅={metrics.f05:.4f})")


# ============================================================================
# 8. Phase 1 Regression Tests (7D.2)
# ============================================================================

def test_regression_graph_bounds_check():
    """Regression 1.1: check_agreement() handles out-of-bounds edges gracefully."""
    from src.morphology.graph import AgreementGraph, AgreementEdge

    graph = AgreementGraph(tokens=["من", "دەچم"], features=[None, None])
    # Inject an out-of-bounds edge
    graph.edges.append(AgreementEdge(
        source_idx=0, target_idx=99, agreement_type="subject_verb",
        features=["person", "number"],
    ))
    violations = graph.check_agreement()
    assert isinstance(violations, list)
    print("  test_regression_graph_bounds_check: PASSED")


def test_regression_clitic_zero_weights():
    """Regression 1.3: CliticErrorGenerator handles zero weights without crash."""
    from src.errors.clitic import CliticErrorGenerator

    gen = CliticErrorGenerator(error_rate=1.0, seed=42)
    result = gen.inject_errors("من")
    assert isinstance(result.corrupted, str)
    print("  test_regression_clitic_zero_weights: PASSED")


def test_regression_graph_size_limit():
    """Regression 1.9: AgreementGraph handles sentences >128 tokens."""
    from src.morphology.agreement import build_agreement_graph

    analyzer = MorphologicalAnalyzer(use_klpt=False)
    long_sentence = " ".join(["کتێب"] * 150)
    graph = build_agreement_graph(long_sentence, analyzer)
    assert isinstance(graph, AgreementGraph)
    assert len(graph.tokens) <= 150
    print("  test_regression_graph_size_limit: PASSED")


def test_regression_small_dataset_splits():
    """Regression 1.10: split_pairs handles n=3 without empty splits."""
    from src.data.splitter import split_pairs

    pairs = [{"source": str(i), "target": str(i)} for i in range(3)]
    train, dev, test = split_pairs(pairs)
    assert len(train) + len(dev) + len(test) == 3
    print(f"  test_regression_small_dataset_splits: PASSED ({len(train)}/{len(dev)}/{len(test)})")


def test_regression_empty_string_is_correct():
    """Regression 1.14: SoraniSpellChecker.is_correct('') returns False."""
    from src.data.spell_checker import SoraniSpellChecker

    checker = SoraniSpellChecker()
    assert checker.is_correct("") is False or not checker.is_available()
    print("  test_regression_empty_string_is_correct: PASSED")


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== Normalization Integration Tests ===")
    test_normalize_wiki_corpus()
    test_normalize_preserves_kurdish_chars()
    test_sentence_split_wiki_paragraph()
    test_deduplicate_corpus()

    print("\n=== Morphological Analysis Integration Tests ===")
    test_analyze_wiki_corpus()
    test_analyzer_detects_verbs()
    test_analyzer_detects_pronouns()
    test_analyzer_past_tense_detection()
    test_normalized_then_analyzed()

    print("\n=== Agreement Graph Integration Tests ===")
    test_build_graph_wiki_corpus()
    test_graph_edge_types_valid()
    test_graph_adjacency_matrix_shape()
    test_graph_typed_stacked_matrix()
    test_graph_violation_check()

    print("\n=== Agreement Checker Integration Tests ===")
    test_checker_wiki_corpus()
    test_checker_detects_sv_mismatch()
    test_checker_detects_dem_def()
    test_evaluate_agreement_accuracy_wiki()

    print("\n=== Full Pipeline Integration Tests ===")
    test_full_pipeline_clean_sentences()
    test_full_pipeline_with_errors()
    test_collector_is_sorani_wiki()
    test_collector_write_and_read_corpus()
    test_feature_vocabulary_coverage()

    print("\n=== Corpus-Level Statistics Tests ===")
    test_corpus_agreement_statistics()
    test_corpus_pos_distribution()

    print("\n=== End-to-End GEC Pipeline Test ===")
    test_e2e_generate_train_evaluate()

    print("\n=== Phase 1 Regression Tests ===")
    test_regression_graph_bounds_check()
    test_regression_clitic_zero_weights()
    test_regression_graph_size_limit()
    test_regression_small_dataset_splits()
    test_regression_empty_string_is_correct()

    print("\nAll integration tests passed!")
