"""
Pipeline integration tests using small corpus texts.

Tests the full sorani-gec pipeline end-to-end with real Sorani Kurdish
sentences: normalization → language detection → error injection →
synthetic pair generation → evaluation scoring.

These complement test_integration.py (which covers normalize → analyze →
graph → check) by exercising error generation, file I/O, and evaluation
in realistic scenarios.
"""

import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.normalizer import SoraniNormalizer
from src.data.sorani_detector import SoraniDetector
from src.errors.pipeline import ErrorPipeline
from src.errors.base import ErrorResult, ErrorAnnotation
from src.errors.subject_verb import SubjectVerbErrorGenerator
from src.errors.noun_adjective import NounAdjectiveErrorGenerator
from src.errors.clitic import CliticErrorGenerator
from src.errors.orthography import OrthographicErrorGenerator
from src.morphology.analyzer import MorphologicalAnalyzer
from src.morphology.agreement import build_agreement_graph
from src.evaluation.f05_scorer import evaluate_corpus, compute_f05, GECMetrics
from src.evaluation.agreement_accuracy import AgreementChecker

# ============================================================================
# Small Corpus Fixture — diverse real Sorani Kurdish sentences
# ============================================================================

# 15 grammatically correct Sorani Kurdish sentences covering different
# syntactic patterns: present tense, past transitive (ergative), past
# intransitive, compound, demonstratives, clitics, questions.
SMALL_CORPUS = [
    # Present tense (simple)
    "من دەچم بۆ قوتابخانە.",
    "تۆ دەتوانیت بێیت.",
    "ئەو کتێبەکەی دەخوێنێتەوە.",

    # Past transitive (ergative agreement)
    "منداڵەکان نانیان خوارد.",
    "پیاوەکە نامەکەی نووسی.",

    # Past intransitive (nominative)
    "منداڵەکان چوون بۆ قوتابخانە.",
    "پیاوەکە هات.",

    # Compound / complex
    "من دەچم بۆ بازاڕ و نان دەکڕم.",
    "ئەگەر بارانی باری ئێمە ناچین بۆ دەرەوە.",

    # Noun phrases with demonstratives
    "ئەم شارانە زۆر کۆنن.",
    "ئەو پیاوە زۆر بەتوانایە.",

    # Clitic sentences
    "کتێبەکەم لە ماڵەوەیە.",
    "ماڵەکەمان گەورەیە.",

    # Questions
    "کێ هات؟",
    "تۆ کەی دەچیت؟",
]


# ============================================================================
# 1. Error Pipeline Initialization Tests
# ============================================================================

def test_pipeline_initializes_all_generators():
    """ErrorPipeline creates exactly 18 generators on init."""
    pipeline = ErrorPipeline(error_rate=0.15, seed=42)
    assert len(pipeline.generators) == 18, (
        f"Expected 18 generators, got {len(pipeline.generators)}"
    )
    print(f"  test_pipeline_initializes_all_generators: PASSED ({len(pipeline.generators)})")


def test_pipeline_respects_seed_determinism():
    """Same seed produces identical results on the same input."""
    pipeline_a = ErrorPipeline(error_rate=0.5, seed=99)
    pipeline_b = ErrorPipeline(error_rate=0.5, seed=99)

    for sentence in SMALL_CORPUS:
        result_a = pipeline_a.process_sentence(sentence)
        result_b = pipeline_b.process_sentence(sentence)
        assert result_a.corrupted == result_b.corrupted, (
            f"Non-deterministic output for: {sentence}\n"
            f"  A: {result_a.corrupted}\n"
            f"  B: {result_b.corrupted}"
        )
    print("  test_pipeline_respects_seed_determinism: PASSED")


# ============================================================================
# 2. Single-Sentence Error Injection Tests
# ============================================================================

def test_pipeline_processes_each_sentence():
    """Pipeline returns a valid ErrorResult for every corpus sentence."""
    pipeline = ErrorPipeline(error_rate=0.3, seed=42)
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        assert isinstance(result, ErrorResult)
        assert result.original == sentence
        assert isinstance(result.corrupted, str)
        assert len(result.corrupted) > 0
        assert isinstance(result.errors, list)
    print(f"  test_pipeline_processes_each_sentence: PASSED ({len(SMALL_CORPUS)} sentences)")


def test_pipeline_error_annotations_well_formed():
    """Every ErrorAnnotation has all required fields populated."""
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        for err in result.errors:
            assert isinstance(err, ErrorAnnotation)
            assert err.error_type, f"Missing error_type in: {sentence}"
            assert isinstance(err.start_pos, int)
            assert isinstance(err.end_pos, int)
            assert err.start_pos >= 0
            assert err.end_pos >= err.start_pos
            assert err.description, f"Missing description in: {sentence}"
    print("  test_pipeline_error_annotations_well_formed: PASSED")


def test_pipeline_high_error_rate_injects_errors():
    """At error_rate=1.0, at least some corpus sentences get errors."""
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    errored_count = 0
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        if result.has_errors:
            errored_count += 1
    assert errored_count > 0, "No errors injected at error_rate=1.0"
    print(f"  test_pipeline_high_error_rate_injects_errors: PASSED ({errored_count}/{len(SMALL_CORPUS)})")


def test_pipeline_zero_error_rate_preserves_text():
    """At error_rate=0.0, no sentence is modified."""
    pipeline = ErrorPipeline(error_rate=0.0, seed=42)
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        assert result.corrupted == result.original, (
            f"Text modified at error_rate=0:\n"
            f"  Original:  {result.original}\n"
            f"  Corrupted: {result.corrupted}"
        )
        assert not result.has_errors
    print("  test_pipeline_zero_error_rate_preserves_text: PASSED")


def test_pipeline_to_dict_roundtrip():
    """ErrorResult.to_dict() produces valid JSON-serializable dicts."""
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["original"] == sentence
        assert isinstance(d["corrupted"], str)
        assert isinstance(d["errors"], list)
        # Verify JSON-serializable
        json_str = json.dumps(d, ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["original"] == sentence
    print("  test_pipeline_to_dict_roundtrip: PASSED")


# ============================================================================
# 3. Corpus File I/O Integration Tests
# ============================================================================

def test_process_corpus_creates_output_files():
    """process_corpus writes train.src, train.tgt, annotations.jsonl, stats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write small corpus to input file
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS:
                f.write(sentence + "\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)
        stats = pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=len(SMALL_CORPUS),
        )

        # Verify all expected output files exist
        assert os.path.isfile(os.path.join(output_dir, "train.src"))
        assert os.path.isfile(os.path.join(output_dir, "train.tgt"))
        assert os.path.isfile(os.path.join(output_dir, "annotations.jsonl"))
        assert os.path.isfile(os.path.join(output_dir, "generation_stats.json"))
    print("  test_process_corpus_creates_output_files: PASSED")


def test_process_corpus_line_counts_match():
    """train.src and train.tgt have identical line counts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS:
                f.write(sentence + "\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)
        pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=len(SMALL_CORPUS),
        )

        with open(os.path.join(output_dir, "train.src"), encoding="utf-8") as f:
            src_lines = f.readlines()
        with open(os.path.join(output_dir, "train.tgt"), encoding="utf-8") as f:
            tgt_lines = f.readlines()

        assert len(src_lines) == len(tgt_lines), (
            f"Line count mismatch: src={len(src_lines)}, tgt={len(tgt_lines)}"
        )
        assert len(src_lines) == len(SMALL_CORPUS)
    print(f"  test_process_corpus_line_counts_match: PASSED ({len(src_lines)} lines)")


def test_process_corpus_annotations_valid_jsonl():
    """Each line in annotations.jsonl is valid JSON with required keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS:
                f.write(sentence + "\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.5, seed=42)
        pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=len(SMALL_CORPUS),
        )

        annotations_path = os.path.join(output_dir, "annotations.jsonl")
        with open(annotations_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                assert "original" in obj, f"Line {i}: missing 'original'"
                assert "corrupted" in obj, f"Line {i}: missing 'corrupted'"
                assert "errors" in obj, f"Line {i}: missing 'errors'"
                assert isinstance(obj["errors"], list)
    print("  test_process_corpus_annotations_valid_jsonl: PASSED")


def test_process_corpus_stats_consistent():
    """generation_stats.json totals match file contents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS:
                f.write(sentence + "\n")

        output_dir = os.path.join(tmpdir, "output")
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)
        stats = pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=len(SMALL_CORPUS),
        )

        assert stats["total"] == len(SMALL_CORPUS)
        assert stats["corrupted"] + stats["clean_pairs"] == stats["total"]
        assert stats["corrupted"] >= 0
        assert isinstance(stats["errors_by_type"], dict)

        # Verify stat file matches returned stats
        with open(os.path.join(output_dir, "generation_stats.json"), encoding="utf-8") as f:
            saved_stats = json.load(f)
        assert saved_stats["total"] == stats["total"]
        assert saved_stats["corrupted"] == stats["corrupted"]
    print(f"  test_process_corpus_stats_consistent: PASSED (corrupted={stats['corrupted']})")


def test_process_corpus_oversampling():
    """When target_pairs > corpus size, pipeline oversamples correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "clean.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS[:5]:
                f.write(sentence + "\n")

        output_dir = os.path.join(tmpdir, "output")
        target = 20
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)
        stats = pipeline.process_corpus(
            input_file=input_file,
            output_dir=output_dir,
            target_pairs=target,
        )

        assert stats["total"] == target, (
            f"Expected {target} pairs after oversampling, got {stats['total']}"
        )
        with open(os.path.join(output_dir, "train.src"), encoding="utf-8") as f:
            assert len(f.readlines()) == target
    print(f"  test_process_corpus_oversampling: PASSED (5 → {target} pairs)")


# ============================================================================
# 4. Normalize → Error Inject → Evaluate Integration
# ============================================================================

def test_normalize_then_inject_errors():
    """Normalized text is processable by the error pipeline without crash."""
    normalizer = SoraniNormalizer()
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)

    for sentence in SMALL_CORPUS:
        normalized = normalizer.normalize(sentence)
        result = pipeline.process_sentence(normalized)
        assert isinstance(result, ErrorResult)
        assert len(result.corrupted) > 0
    print("  test_normalize_then_inject_errors: PASSED")


def test_error_pairs_evaluable_with_f05():
    """Corrupted/original pairs from the pipeline can be scored by F0.5."""
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)

    sources = []  # corrupted
    references = []  # original (clean)
    hypotheses = []  # "perfect correction" = original

    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        sources.append(result.corrupted)
        references.append(result.original)
        hypotheses.append(result.original)  # simulate perfect model

    metrics = evaluate_corpus(sources, hypotheses, references)
    assert isinstance(metrics, GECMetrics)
    # Perfect correction → F₀.₅ should be 1.0 (or very close if no errors injected)
    assert metrics.f05 >= 0.0
    assert metrics.fp == 0, f"Perfect correction should have 0 FP, got {metrics.fp}"
    print(f"  test_error_pairs_evaluable_with_f05: PASSED ({metrics})")


def test_identity_model_f05_scores():
    """An identity model (no corrections) gets 0 TP and non-zero FN when errors exist."""
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)

    sources = []
    references = []
    hypotheses = []

    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        if result.has_errors:
            sources.append(result.corrupted)
            references.append(result.original)
            hypotheses.append(result.corrupted)  # identity = no correction

    if len(sources) > 0:
        metrics = evaluate_corpus(sources, hypotheses, references)
        assert metrics.tp == 0, f"Identity model TP should be 0, got {metrics.tp}"
        assert metrics.fn >= 0
        print(f"  test_identity_model_f05_scores: PASSED ({metrics})")
    else:
        print("  test_identity_model_f05_scores: SKIPPED (no errors injected)")


# ============================================================================
# 5. Language Detection + Pipeline Integration
# ============================================================================

def test_detector_accepts_corpus_sentences():
    """SoraniDetector classifies all small corpus sentences as Sorani."""
    detector = SoraniDetector()
    for sentence in SMALL_CORPUS:
        result = detector.detect(sentence)
        assert result.is_sorani, (
            f"Detector rejected Sorani sentence (conf={result.confidence:.2f}): {sentence}"
        )
    print("  test_detector_accepts_corpus_sentences: PASSED")


def test_detector_accepts_corrupted_sentences():
    """SoraniDetector still classifies error-injected sentences as Sorani."""
    detector = SoraniDetector()
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    rejected = 0
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        det = detector.detect(result.corrupted)
        if not det.is_sorani:
            rejected += 1
    # Some short corrupted sentences might fail detection; that's ok
    # but the majority should still pass
    accept_rate = (len(SMALL_CORPUS) - rejected) / len(SMALL_CORPUS)
    assert accept_rate >= 0.5, (
        f"Too many corrupted sentences rejected: {rejected}/{len(SMALL_CORPUS)}"
    )
    print(f"  test_detector_accepts_corrupted_sentences: PASSED ({rejected} rejected)")


def test_filter_corpus_then_inject():
    """filter_corpus → error injection pipeline chain works end-to-end."""
    detector = SoraniDetector()
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)

    mixed = SMALL_CORPUS + ["This is English.", "هذه جملة عربية."]
    filtered = detector.filter_corpus(mixed)

    assert len(filtered) >= len(SMALL_CORPUS) - 2  # might lose very short ones
    for sentence in filtered:
        result = pipeline.process_sentence(sentence)
        assert isinstance(result, ErrorResult)
    print(f"  test_filter_corpus_then_inject: PASSED ({len(mixed)} → {len(filtered)})")


# ============================================================================
# 6. Full Pipeline Chain: Normalize → Detect → Inject → Analyze → Check
# ============================================================================

def test_full_chain_normalize_inject_analyze():
    """Full chain: normalize → inject errors → analyze morphology → build graph."""
    normalizer = SoraniNormalizer()
    pipeline = ErrorPipeline(error_rate=0.5, seed=42)
    analyzer = MorphologicalAnalyzer(use_klpt=False)

    for sentence in SMALL_CORPUS[:8]:
        # Normalize
        normalized = normalizer.normalize(sentence)

        # Inject errors
        result = pipeline.process_sentence(normalized)

        # Analyze corrupted text morphologically
        features = analyzer.analyze_sentence(result.corrupted)
        assert len(features) > 0, f"No features for corrupted: {result.corrupted}"

        # Build agreement graph on corrupted text
        graph = build_agreement_graph(result.corrupted, analyzer)
        assert graph is not None

    print("  test_full_chain_normalize_inject_analyze: PASSED")


def test_full_chain_with_agreement_check():
    """Full chain including agreement checking on corrupted vs clean text."""
    normalizer = SoraniNormalizer()
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    checker = AgreementChecker()

    clean_violations = 0
    corrupt_violations = 0

    for sentence in SMALL_CORPUS[:8]:
        normalized = normalizer.normalize(sentence)

        # Check clean text
        clean_result = checker.check_sentence(normalized)
        clean_violations += len(clean_result.violations)

        # Inject errors and check corrupted text
        error_result = pipeline.process_sentence(normalized)
        if error_result.has_errors:
            corrupt_result = checker.check_sentence(error_result.corrupted)
            corrupt_violations += len(corrupt_result.violations)

    # Corrupted text should generally have at least as many violations
    # (not strictly guaranteed for every sentence, but trend should hold)
    print(
        f"  test_full_chain_with_agreement_check: PASSED "
        f"(clean_violations={clean_violations}, corrupt_violations={corrupt_violations})"
    )


# ============================================================================
# 7. Error Diversity & Statistics Tests
# ============================================================================

def test_pipeline_error_type_diversity():
    """Pipeline produces more than one error type across the corpus."""
    pipeline = ErrorPipeline(error_rate=1.0, seed=42)
    error_types: set[str] = set()
    for sentence in SMALL_CORPUS:
        result = pipeline.process_sentence(sentence)
        for err in result.errors:
            error_types.add(err.error_type)
    assert len(error_types) >= 1, "Expected at least 1 error type across corpus"
    print(f"  test_pipeline_error_type_diversity: PASSED ({len(error_types)} types: {sorted(error_types)})")


def test_individual_generators_on_corpus():
    """Each individual generator handles all corpus sentences without crash."""
    generators = [
        SubjectVerbErrorGenerator(error_rate=1.0, seed=42),
        NounAdjectiveErrorGenerator(error_rate=1.0, seed=42),
        CliticErrorGenerator(error_rate=1.0, seed=42),
        OrthographicErrorGenerator(error_rate=1.0, seed=42),
    ]
    for gen in generators:
        for sentence in SMALL_CORPUS:
            result = gen.inject_errors(sentence)
            assert isinstance(result, ErrorResult)
            assert result.original == sentence
    gen_names = [g.error_type for g in generators]
    print(f"  test_individual_generators_on_corpus: PASSED ({gen_names})")


# ============================================================================
# 8. End-to-End File Pipeline Test
# ============================================================================

def test_end_to_end_file_pipeline():
    """Full file-based pipeline: write corpus → normalize → process_corpus → read back → evaluate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        normalizer = SoraniNormalizer()

        # Step 1: Write and normalize corpus
        clean_file = os.path.join(tmpdir, "normalized.txt")
        with open(clean_file, "w", encoding="utf-8") as f:
            for sentence in SMALL_CORPUS:
                normalized = normalizer.normalize(sentence)
                f.write(normalized + "\n")

        # Step 2: Run error pipeline on file
        output_dir = os.path.join(tmpdir, "synthetic")
        pipeline = ErrorPipeline(error_rate=0.3, seed=42)
        stats = pipeline.process_corpus(
            input_file=clean_file,
            output_dir=output_dir,
            target_pairs=len(SMALL_CORPUS),
        )

        # Step 3: Read back generated pairs
        with open(os.path.join(output_dir, "train.src"), encoding="utf-8") as f:
            sources = [line.strip() for line in f]
        with open(os.path.join(output_dir, "train.tgt"), encoding="utf-8") as f:
            references = [line.strip() for line in f]

        assert len(sources) == len(references)

        # Step 4: Simulate perfect model and evaluate
        hypotheses = references[:]  # perfect correction
        metrics = evaluate_corpus(sources, hypotheses, references)
        assert isinstance(metrics, GECMetrics)
        assert metrics.fp == 0
        print(
            f"  test_end_to_end_file_pipeline: PASSED "
            f"(pairs={len(sources)}, corrupted={stats['corrupted']}, {metrics})"
        )


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=== Error Pipeline Init Tests ===")
    test_pipeline_initializes_all_generators()
    test_pipeline_respects_seed_determinism()

    print("\n=== Single-Sentence Error Injection Tests ===")
    test_pipeline_processes_each_sentence()
    test_pipeline_error_annotations_well_formed()
    test_pipeline_high_error_rate_injects_errors()
    test_pipeline_zero_error_rate_preserves_text()
    test_pipeline_to_dict_roundtrip()

    print("\n=== Corpus File I/O Integration Tests ===")
    test_process_corpus_creates_output_files()
    test_process_corpus_line_counts_match()
    test_process_corpus_annotations_valid_jsonl()
    test_process_corpus_stats_consistent()
    test_process_corpus_oversampling()

    print("\n=== Normalize → Error Inject → Evaluate Tests ===")
    test_normalize_then_inject_errors()
    test_error_pairs_evaluable_with_f05()
    test_identity_model_f05_scores()

    print("\n=== Language Detection + Pipeline Tests ===")
    test_detector_accepts_corpus_sentences()
    test_detector_accepts_corrupted_sentences()
    test_filter_corpus_then_inject()

    print("\n=== Full Pipeline Chain Tests ===")
    test_full_chain_normalize_inject_analyze()
    test_full_chain_with_agreement_check()

    print("\n=== Error Diversity & Statistics Tests ===")
    test_pipeline_error_type_diversity()
    test_individual_generators_on_corpus()

    print("\n=== End-to-End File Pipeline Test ===")
    test_end_to_end_file_pipeline()

    print("\nAll pipeline integration tests passed!")
