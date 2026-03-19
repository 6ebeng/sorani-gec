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
from ..data.spell_checker import SoraniSpellChecker

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
        
        # Initialize all generators
        self.generators: list[BaseErrorGenerator] = [
            SubjectVerbErrorGenerator(error_rate=error_rate, seed=seed),
            NounAdjectiveErrorGenerator(error_rate=error_rate, seed=seed + 1),
            CliticErrorGenerator(error_rate=error_rate, seed=seed + 2),
            TenseAgreementErrorGenerator(error_rate=error_rate, seed=seed + 3),
            CaseRoleErrorGenerator(error_rate=error_rate, seed=seed + 4),
            DialectalParticipleErrorGenerator(error_rate=error_rate, seed=seed + 5),
            RelativeClauseErrorGenerator(error_rate=error_rate, seed=seed + 6),
            AdversativeConnectorErrorGenerator(error_rate=error_rate, seed=seed + 7),
            ParticipleSwapErrorGenerator(error_rate=error_rate, seed=seed + 8),
            OrthographicErrorGenerator(error_rate=error_rate, seed=seed + 9),
            NegativeConcordErrorGenerator(error_rate=error_rate, seed=seed + 10),
            VocativeImperativeErrorGenerator(error_rate=error_rate, seed=seed + 11),
            ConditionalAgreementErrorGenerator(error_rate=error_rate, seed=seed + 12),
            AdverbVerbTenseErrorGenerator(error_rate=error_rate, seed=seed + 13),
            PrepositionFusionErrorGenerator(error_rate=error_rate, seed=seed + 14),
            DemonstrativeContractionErrorGenerator(error_rate=error_rate, seed=seed + 15),
            QuantifierAgreementErrorGenerator(error_rate=error_rate, seed=seed + 16),
            PossessiveCliticErrorGenerator(error_rate=error_rate, seed=seed + 17),
            PoliteImperativeErrorGenerator(error_rate=error_rate, seed=seed + 18),
        ]
        
        logger.info("Initialized pipeline with %d error generators", len(self.generators))
    
    def process_sentence(self, sentence: str) -> ErrorResult:
        """Apply error generators to a single sentence.
        
        Randomly selects 1-2 generators to apply per sentence.
        """
        # Randomly select generators for this sentence
        n_generators = min(
            self.rng.randint(1, self.max_errors_per_sentence),
            len(self.generators),
        )
        selected = self.rng.sample(self.generators, n_generators)
        
        current_text = sentence
        all_errors = []
        
        for generator in selected:
            result = generator.inject_errors(current_text)
            if result.has_errors:
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
    ) -> dict:
        """Process a clean corpus file and generate synthetic parallel data.
        
        Args:
            input_file: Path to clean text file (one sentence per line).
            output_dir: Directory to save output files.
            target_pairs: Target number of parallel pairs.
            corruption_ratio: Fraction of sentences to corrupt (rest stay clean→clean).
            
        Returns:
            Statistics dictionary.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read clean sentences
        with open(input_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        logger.info("Loaded %d clean sentences", len(sentences))
        
        # Preprocessing: spell-check clean sentences if available
        spell_checker = SoraniSpellChecker()
        if hasattr(spell_checker, "correct_sentence") and callable(spell_checker.correct_sentence):
            sentences = [spell_checker.correct_sentence(s) for s in sentences]
            logger.info("Spell-checked clean corpus via SoraniSpellChecker")
        else:
            logger.info("SoraniSpellChecker.correct_sentence() not available — skipping spell check")
        
        if len(sentences) < target_pairs:
            # Oversample if needed
            multiplier = (target_pairs // len(sentences)) + 1
            sentences = (sentences * multiplier)[:target_pairs]
            logger.info("Oversampled to %d sentences", len(sentences))
        else:
            sentences = sentences[:target_pairs]
        
        # Process sentences
        results = []
        stats = {
            "total": len(sentences),
            "corrupted": 0,
            "clean_pairs": 0,
            "errors_by_type": {},
        }
        
        for sentence in tqdm(sentences, desc="Generating errors"):
            if self.rng.random() < corruption_ratio:
                result = self.process_sentence(sentence)
                if result.has_errors:
                    stats["corrupted"] += 1
                    for err in result.errors:
                        stats["errors_by_type"][err.error_type] = \
                            stats["errors_by_type"].get(err.error_type, 0) + 1
                else:
                    stats["clean_pairs"] += 1
            else:
                result = ErrorResult(original=sentence, corrupted=sentence, errors=[])
                stats["clean_pairs"] += 1
            
            results.append(result)
        
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
