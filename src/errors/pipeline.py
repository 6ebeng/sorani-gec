"""
Synthetic Error Generation Pipeline

Orchestrates all error generators to produce a synthetic error-annotated dataset
of (clean, corrupted) Sorani Kurdish sentence pairs with error annotations.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .base import BaseErrorGenerator, ErrorResult
from .subject_verb import SubjectVerbErrorGenerator
from .noun_adjective import NounAdjectiveErrorGenerator
from .clitic import CliticErrorGenerator
from .tense_agreement import TenseAgreementErrorGenerator
from .syntax_roles import CaseRoleErrorGenerator
from .dialectal import DialectalParticipleErrorGenerator
from .relative_clause import RelativeClauseErrorGenerator
from .adversative import AdversativeConnectorErrorGenerator
from .participle_swap import ParticipleSwapErrorGenerator
from .orthography import OrthographicErrorGenerator
from .negative_concord import NegativeConcordErrorGenerator
from .vocative_imperative import VocativeImperativeErrorGenerator
from .conditional_agreement import ConditionalAgreementErrorGenerator
from .adverb_verb_tense import AdverbVerbTenseErrorGenerator
from .preposition_fusion import PrepositionFusionErrorGenerator
from .demonstrative_contraction import DemonstrativeContractionErrorGenerator
from .quantifier_agreement import QuantifierAgreementErrorGenerator
from .possessive_clitic import PossessiveCliticErrorGenerator
from .polite_imperative import PoliteImperativeErrorGenerator
from .spelling_confusion import SpellingConfusionErrorGenerator
from .word_order import WordOrderErrorGenerator
from .whitespace_error import WhitespaceErrorGenerator
from .punctuation_error import PunctuationErrorGenerator
from .cross_clause_agreement import CrossClauseAgreementErrorGenerator
from ..data.tokenize import sorani_word_tokenize
from .morpheme_order import MorphemeOrderErrorGenerator
from ..data.spell_checker import SoraniSpellChecker
from ..morphology.analyzer import MorphologicalAnalyzer

logger = logging.getLogger(__name__)


class ErrorPipeline:
    """Pipeline that applies multiple error generators to produce synthetic data."""
    
    def __init__(
        self,
        error_rate: float = 0.15,
        max_errors_per_sentence: int = 2,
        seed: int = 42,
    ):
        self.error_rate = error_rate
        self.max_errors_per_sentence = max_errors_per_sentence
        self.rng = random.Random(seed)
        
        # Shared morphological analyzer for dynamic verb/noun recognition
        analyzer = MorphologicalAnalyzer()

        # Initialize all generators with frequency weights.
        # Weights reflect realistic error frequency in learner data:
        # agreement/clitic errors are far more common than stylistic ones.
        self.generators: list[BaseErrorGenerator] = [
            SubjectVerbErrorGenerator(error_rate=error_rate, seed=seed, analyzer=analyzer),
            NounAdjectiveErrorGenerator(error_rate=error_rate, seed=seed + 1, analyzer=analyzer),
            CliticErrorGenerator(error_rate=error_rate, seed=seed + 2, analyzer=analyzer),
            TenseAgreementErrorGenerator(error_rate=error_rate, seed=seed + 3, analyzer=analyzer),
            CaseRoleErrorGenerator(error_rate=error_rate, seed=seed + 4),
            DialectalParticipleErrorGenerator(error_rate=error_rate, seed=seed + 5),
            RelativeClauseErrorGenerator(error_rate=error_rate, seed=seed + 6),
            AdversativeConnectorErrorGenerator(error_rate=error_rate, seed=seed + 7),
            ParticipleSwapErrorGenerator(error_rate=error_rate, seed=seed + 8),
            OrthographicErrorGenerator(error_rate=error_rate, seed=seed + 9),
            NegativeConcordErrorGenerator(error_rate=error_rate, seed=seed + 10),
            VocativeImperativeErrorGenerator(error_rate=error_rate, seed=seed + 11),
            ConditionalAgreementErrorGenerator(error_rate=error_rate, seed=seed + 12, analyzer=analyzer),
            AdverbVerbTenseErrorGenerator(error_rate=error_rate, seed=seed + 13),
            PrepositionFusionErrorGenerator(error_rate=error_rate, seed=seed + 14),
            DemonstrativeContractionErrorGenerator(error_rate=error_rate, seed=seed + 15),
            QuantifierAgreementErrorGenerator(error_rate=error_rate, seed=seed + 16),
            PossessiveCliticErrorGenerator(error_rate=error_rate, seed=seed + 17),
            PoliteImperativeErrorGenerator(error_rate=error_rate, seed=seed + 18),
            SpellingConfusionErrorGenerator(error_rate=error_rate, seed=seed + 19),
            WordOrderErrorGenerator(error_rate=error_rate, seed=seed + 20),
            WhitespaceErrorGenerator(error_rate=error_rate, seed=seed + 21),
            PunctuationErrorGenerator(error_rate=error_rate, seed=seed + 22),
            CrossClauseAgreementErrorGenerator(error_rate=error_rate, seed=seed + 23),
            MorphemeOrderErrorGenerator(error_rate=error_rate, seed=seed + 24),
        ]
        # Frequency weights: higher = more likely to be selected.
        # Core agreement errors (subject-verb, noun-adj, clitic, tense) are
        # weighted ~3x heavier than rare stylistic patterns.
        self.generator_weights: list[float] = [
            3.0,  # subject_verb — very common
            3.0,  # noun_adjective — very common
            2.5,  # clitic — common
            2.5,  # tense_agreement — common
            1.0,  # syntax_roles
            0.5,  # dialectal
            1.0,  # relative_clause
            0.5,  # adversative
            0.5,  # participle_swap
            2.0,  # orthography — common in written text
            1.0,  # negative_concord
            0.5,  # vocative_imperative
            1.5,  # conditional_agreement
            1.0,  # adverb_verb_tense
            0.5,  # preposition_fusion
            1.0,  # demonstrative_contraction
            1.5,  # quantifier_agreement
            1.5,  # possessive_clitic
            0.5,  # polite_imperative
            2.0,  # spelling_confusion — common in informal text
            0.5,  # word_order — less common (SOV is flexible)
            1.5,  # whitespace — common in Arabic-script text
            1.0,  # punctuation — moderately common
            0.5,  # cross_clause_agreement — complex sentences only
            0.5,  # morpheme_order — relatively rare
        ]

        assert len(self.generators) == len(self.generator_weights), (
            f"Generator count ({len(self.generators)}) != weight count "
            f"({len(self.generator_weights)}). Add/remove weights to match."
        )
        
        logger.info("Initialized pipeline with %d error generators", len(self.generators))
    
    def process_sentence(self, sentence: str) -> ErrorResult:
        """Apply error generators to a single sentence.
        
        Randomly selects 1-2 generators using frequency-weighted sampling.
        """
        # Randomly select generators using frequency weights
        n_generators = min(
            self.rng.randint(1, self.max_errors_per_sentence),
            len(self.generators),
        )
        # random.choices with weights (with replacement) then deduplicate
        # to avoid applying the same generator twice
        candidates = self.rng.choices(
            self.generators, weights=self.generator_weights, k=n_generators * 2,
        )
        seen = set()
        selected = []
        for gen in candidates:
            gen_id = id(gen)
            if gen_id not in seen:
                seen.add(gen_id)
                selected.append(gen)
                if len(selected) >= n_generators:
                    break
        
        current_text = sentence
        all_errors = []
        modified_word_indices: set[int] = set()
        # 6B.7: Track char-level modified ranges for sub-word edits
        modified_char_ranges: list[tuple[int, int]] = []
        
        for generator in selected:
            result = generator.inject_errors(
                current_text,
                skip_word_indices=modified_word_indices if modified_word_indices else None,
            )
            if result.has_errors:
                # Track which word indices changed (double-flip avoidance)
                orig_words = sorani_word_tokenize(current_text)
                corr_words = sorani_word_tokenize(result.corrupted)
                for i in range(min(len(orig_words), len(corr_words))):
                    if orig_words[i] != corr_words[i]:
                        modified_word_indices.add(i)
                # If word count changed, mark the extra region too
                if len(corr_words) != len(orig_words):
                    modified_word_indices.update(
                        range(min(len(orig_words), len(corr_words)),
                              max(len(orig_words), len(corr_words)))
                    )
                # 6B.7: Record char-level ranges from error annotations
                for err in result.errors:
                    modified_char_ranges.append(
                        (err.start_pos, err.end_pos)
                    )

                current_text = result.corrupted
                # Errors from this generator have positions relative to the
                # text that was fed into this generator — which is already
                # the cumulative corrupted output.  Record them as-is;
                # they describe spans in the generator's input (not the
                # original sentence), but the final corrupted text is what
                # matters for the (source, target) training pair.
                all_errors.extend(result.errors)
        
        return ErrorResult(
            original=sentence,
            corrupted=current_text,
            errors=all_errors,
        )
    
    def process_corpus(
        self,
        input_file: str,
        output_dir: str,
        target_pairs: int = 50000,
        corruption_ratio: float = 0.7,
        spell_check_clean: bool = False,
        validate_errors: bool = False,
    ) -> dict:
        """Process a clean corpus file and generate synthetic parallel data.
        
        Args:
            input_file: Path to clean text file (one sentence per line).
            output_dir: Directory to save output files.
            target_pairs: Target number of parallel pairs.
            corruption_ratio: Fraction of sentences to corrupt (rest stay clean→clean).
            spell_check_clean: If True, spell-check clean sentences before error injection.
            validate_errors: If True, reject error pairs where all injected error
                tokens are valid dictionary words (CRIT-6). Such pairs add noise
                because the "error" is indistinguishable from correct Sorani.
            
        Returns:
            Statistics dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read clean sentences
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        logger.info("Loaded %d clean sentences", len(sentences))
        
        # Truncate/oversample to target BEFORE expensive spell-check
        if len(sentences) < target_pairs:
            multiplier = (target_pairs // len(sentences)) + 1
            sentences = (sentences * multiplier)[:target_pairs]
            logger.info("Oversampled to %d sentences", len(sentences))
        else:
            sentences = sentences[:target_pairs]
        
        # Preprocessing: spell-check clean sentences if requested
        if spell_check_clean:
            spell_checker = SoraniSpellChecker()
            if hasattr(spell_checker, "correct_sentence") and callable(spell_checker.correct_sentence):
                sentences = [spell_checker.correct_sentence(s) for s in sentences]
                logger.info("Spell-checked %d clean sentences via SoraniSpellChecker", len(sentences))
            else:
                logger.info("SoraniSpellChecker.correct_sentence() not available — skipping spell check")
        
        # Process sentences
        results = []
        stats = {
            "total": len(sentences),
            "corrupted": 0,
            "clean_pairs": 0,
            "errors_by_type": {},
            "validation_rejected": 0,
        }

        # CRIT-6: Spell-check validation filter
        validator = None
        if validate_errors:
            validator = SoraniSpellChecker()
            if not validator.is_available():
                logger.warning(
                    "Spell checker not available — skipping error validation"
                )
                validator = None
            else:
                logger.info("Error validation enabled via SoraniSpellChecker")
        
        import itertools
        from tqdm import tqdm
        
        target_corrupted = int(target_pairs * corruption_ratio)
        target_clean = target_pairs - target_corrupted
        
        self.rng.shuffle(sentences)
        iter_sentences = itertools.cycle(enumerate(sentences))
        
        with tqdm(total=target_pairs, desc="Generating errors") as pbar:
            while len(results) < target_pairs:
                idx, sentence = next(iter_sentences)
                source_id = str(idx)
                
                if stats["corrupted"] < target_corrupted and stats["clean_pairs"] < target_clean:
                    want_corrupt = self.rng.random() < corruption_ratio
                elif stats["corrupted"] < target_corrupted:
                    want_corrupt = True
                else:
                    want_corrupt = False
                    
                if want_corrupt:
                    result = self.process_sentence(sentence)
                    result.source_id = source_id
                    if result.has_errors:
                        # CRIT-6: Validate that injected errors aren't valid words
                        if validator is not None:
                            orig_words = set(result.original.split())
                            corr_words = set(result.corrupted.split())
                            new_tokens = corr_words - orig_words
                            if new_tokens and all(validator.is_correct(w) for w in new_tokens):
                                stats["validation_rejected"] += 1
                                continue  # Reject, try another sentence
                        
                        stats["corrupted"] += 1
                        for err in result.errors:
                            stats["errors_by_type"][err.error_type] = stats["errors_by_type"].get(err.error_type, 0) + 1
                        results.append(result)
                        pbar.update(1)
                    else:
                        continue # Failed to add error, try next sentence (drops source==target)
                else:
                    stats["clean_pairs"] += 1
                    result = ErrorResult(
                        original=sentence, corrupted=sentence, errors=[],
                        source_id=source_id,
                    )
                    results.append(result)
                    pbar.update(1)
        
        self.rng.shuffle(results)
        # Save outputs
        # 1. GEC training pairs (source → target)
        src_file = output_path / "train.src"  # corrupted
        tgt_file = output_path / "train.tgt"  # clean
        
        with open(src_file, "w", encoding="utf-8") as fsrc, \
             open(tgt_file, "w", encoding="utf-8") as ftgt:
            for result in results:
                fsrc.write(result.corrupted + "\n")
                ftgt.write(result.original + "\n")
        
        # 2. Detailed annotations (JSON)
        annotations_file = output_path / "annotations.jsonl"
        with open(annotations_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        
        # 3. Stats
        stats_file = output_path / "generation_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info("Generated %d corrupted + %d clean pairs", stats['corrupted'], stats['clean_pairs'])
        logger.info("Errors by type: %s", stats['errors_by_type'])
        logger.info("Saved to %s", output_path)
        
        return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    pipeline = ErrorPipeline(error_rate=0.15, seed=42)
    
    # Example usage
    test_sentence = "من دەچم بۆ قوتابخانە"  # "I go to school"
    result = pipeline.process_sentence(test_sentence)
    
    print(f"Original:  {result.original}")
    print(f"Corrupted: {result.corrupted}")
    print(f"Errors:    {[e.description for e in result.errors]}")
