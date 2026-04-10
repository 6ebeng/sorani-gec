"""
Corpus Catalog — Category-Aware Corpus Management

Maps source documents (academic theses, dissertations, books) to academic
discipline categories and provides balanced sampling so that the final
training corpus draws evenly across majors.

Categories are broad enough to capture meaningful diversity but narrow
enough that each category covers a distinct register and vocabulary.
"""

import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Academic discipline categories
# -------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "linguistics",
    "literature",
    "history",
    "education",
    "law",
    "political_science",
    "social_sciences",
    "sciences",
    "engineering",
    "computer_science",
    "economics",
    "islamic_studies",
    "media",
    "general",
]

# Keywords (in Sorani, Kurdish Latin, and English) that hint at each
# category.  Used only when no explicit mapping is provided; the user
# can always override via a catalog JSON file.
_CATEGORY_HINTS: dict[str, list[str]] = {
    "linguistics": [
        "زمان", "ڕێزمان", "ڕستەساز", "مۆرفۆلۆج", "سینتاکس",
        "سیمانتیک", "فۆنۆلۆج", "وشەساز", "زمانەوان",
        "morpho", "syntax", "linguistic", "grammar", "phonolog",
    ],
    "literature": [
        "ئەدەب", "شیعر", "ڕۆمان", "چیرۆک", "نووسەر", "ڕەخنە",
        "literature", "poetry", "novel", "narrative",
    ],
    "history": [
        "مێژوو", "تاریخ", "شاری", "شوێن", "history", "historical",
        "archeolog", "ancient",
    ],
    "education": [
        "پەروەردە", "خوێندن", "فێرکردن", "مامۆستا", "قوتابخانە",
        "education", "pedagog", "curricul", "teaching",
    ],
    "law": [
        "یاسا", "دادگا", "دەستوور", "تاوان", "مافی",
        "law", "legal", "constitution", "criminal", "justice",
    ],
    "political_science": [
        "سیاس", "حزب", "دیمۆکراس", "حکومەت", "فیدراڵ",
        "politic", "democra", "govern", "federal",
    ],
    "social_sciences": [
        "کۆمەڵ", "دەروون", "ئابووری", "فەلسەفە",
        "social", "psycholog", "sociol", "anthropol", "philosophy",
    ],
    "sciences": [
        "فیزیا", "کیمیا", "بایۆلۆج", "ژینگە", "بیرکاری",
        "physics", "chemistr", "biolog", "environment", "math",
    ],
    "engineering": [
        "ئەندازیار", "تەلارساز", "شارساز", "ئاو", "وزە",
        "engineer", "architect", "civil", "mechanical", "electric",
    ],
    "computer_science": [
        "کۆمپیوتەر", "زانیاری", "بەرنامەساز", "تەکنەلۆج",
        "computer", "software", "algorithm", "data", "network", "AI",
    ],
    "economics": [
        "ئابووری", "بازار", "بانک", "بازرگانی",
        "econom", "market", "financ", "trade", "business",
    ],
    "islamic_studies": [
        "ئیسلام", "قورئان", "فیقه", "شەریعە", "تەفسیر",
        "islam", "quran", "sharia", "theology",
    ],
    "media": [
        "میدیا", "ڕۆژنامە", "ڕاگەیاندن", "تەلەفزیۆن",
        "media", "journal", "broadcast", "press",
    ],
}

# -------------------------------------------------------------------------
# KTC (Kurdish Textbooks Corpus) category mapping
# Maps KTC directory names → our CATEGORIES.
# Source: https://github.com/KurdishBLARK/KTC
# -------------------------------------------------------------------------

KTC_CATEGORY_MAP: dict[str, str] = {
    "economy": "economics",
    "genocide": "history",
    "geography": "social_sciences",
    "history": "history",
    "human-rights": "law",
    "kurdish": "linguistics",
    "kurdology": "linguistics",
    "philosophy": "social_sciences",
    "physics": "sciences",
    "social-study": "social_sciences",
    "sociology": "social_sciences",
    "theology": "islamic_studies",
}


@dataclass
class SourceDocument:
    """Metadata for a single source document."""

    filename: str
    category: str
    sentence_count: int = 0
    char_count: int = 0
    title: str = ""


@dataclass
class CatalogStats:
    """Summary statistics for a categorized corpus."""

    total_sentences: int = 0
    total_documents: int = 0
    per_category: dict[str, int] = field(default_factory=dict)
    per_document: dict[str, int] = field(default_factory=dict)


class CorpusCatalog:
    """Category-aware corpus manager.

    Reads a catalog file (JSON) that maps source documents to academic
    categories, counts sentences per source, and provides balanced
    sampling across categories.

    Catalog JSON format::

        {
          "documents": [
            {
              "filename": "linguistics_thesis_01.txt",
              "category": "linguistics",
              "title": "Syntax of Kurdish verbs"
            },
            ...
          ]
        }

    If no catalog file is provided, categories are inferred from
    filename keywords using ``_CATEGORY_HINTS``.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        catalog_path: Optional[str | Path] = None,
        seed: int = 42,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.seed = seed
        self._rng = random.Random(seed)

        self.documents: list[SourceDocument] = []
        self._sentences_by_doc: dict[str, list[str]] = {}
        self._sentences_by_category: dict[str, list[str]] = {}

        if catalog_path and Path(catalog_path).exists():
            self._load_catalog(Path(catalog_path))
        else:
            self._auto_catalog()

    # ------------------------------------------------------------------
    # Catalog loading
    # ------------------------------------------------------------------

    def _load_catalog(self, catalog_path: Path) -> None:
        """Load explicit category mapping from JSON."""
        with open(catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("documents", []):
            fn = entry["filename"]
            cat = entry.get("category", "general")
            if cat not in CATEGORIES:
                logger.warning("Unknown category '%s' for %s — using 'general'", cat, fn)
                cat = "general"
            title = entry.get("title", fn)
            self.documents.append(SourceDocument(
                filename=fn, category=cat, title=title,
            ))

        logger.info("Loaded catalog with %d documents from %s", len(self.documents), catalog_path)

    def _auto_catalog(self) -> None:
        """Infer categories from filenames in corpus_dir."""
        if not self.corpus_dir.exists():
            logger.warning("Corpus directory %s does not exist", self.corpus_dir)
            return

        for txt_file in sorted(self.corpus_dir.glob("*.txt")):
            if txt_file.name.startswith("."):
                continue
            cat = self._infer_category(txt_file.name)
            self.documents.append(SourceDocument(
                filename=txt_file.name, category=cat, title=txt_file.stem,
            ))

        logger.info(
            "Auto-cataloged %d documents from %s",
            len(self.documents), self.corpus_dir,
        )

    @staticmethod
    def _infer_category(filename: str) -> str:
        """Guess category from filename keywords."""
        name_lower = filename.lower()
        best_cat = "general"
        best_score = 0

        for cat, hints in _CATEGORY_HINTS.items():
            score = sum(1 for h in hints if h in name_lower or h in filename)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    # ------------------------------------------------------------------
    # Sentence loading
    # ------------------------------------------------------------------

    def load_sentences(self) -> CatalogStats:
        """Read all documents and count sentences per category.

        Sentences are stored in memory grouped by document and by
        category for balanced sampling.

        Returns per-category and per-document statistics.
        """
        self._sentences_by_doc.clear()
        self._sentences_by_category.clear()

        for cat in CATEGORIES:
            self._sentences_by_category[cat] = []

        for doc in self.documents:
            fpath = self.corpus_dir / doc.filename
            if not fpath.exists():
                logger.warning("Document not found: %s", fpath)
                continue

            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = [line.strip() for line in f if line.strip()]

            doc.sentence_count = len(lines)
            doc.char_count = sum(len(line) for line in lines)
            self._sentences_by_doc[doc.filename] = lines

            if doc.category not in self._sentences_by_category:
                self._sentences_by_category[doc.category] = []
            self._sentences_by_category[doc.category].extend(lines)

        stats = self._compute_stats()
        logger.info(
            "Loaded %d sentences from %d documents across %d categories",
            stats.total_sentences,
            stats.total_documents,
            len([c for c, n in stats.per_category.items() if n > 0]),
        )
        return stats

    def _compute_stats(self) -> CatalogStats:
        """Build summary statistics from loaded data."""
        stats = CatalogStats()
        stats.total_documents = len([d for d in self.documents if d.sentence_count > 0])
        for doc in self.documents:
            stats.per_document[doc.filename] = doc.sentence_count
        for cat, sents in self._sentences_by_category.items():
            if sents:
                stats.per_category[cat] = len(sents)
        stats.total_sentences = sum(stats.per_category.values())
        return stats

    # ------------------------------------------------------------------
    # Balanced sampling
    # ------------------------------------------------------------------

    def balanced_sample(
        self,
        target_sentences: int = 50000,
        min_per_category: int = 100,
    ) -> dict[str, list[str]]:
        """Draw sentences evenly across categories up to *target_sentences*.

        Strategy:
        1. Determine active categories (those with ≥ *min_per_category*
           sentences after loading).
        2. Compute an equal quota per category: ``target / n_active``.
        3. Small categories that have fewer sentences than the quota
           contribute all of their sentences; the shortfall is
           redistributed among larger categories.
        4. Within each category, sentences are shuffled (deterministic
           via *seed*) and the first *quota* are taken.

        Returns a dict mapping category name → list of sampled sentences.
        """
        if not self._sentences_by_category:
            self.load_sentences()

        # Active categories with enough sentences
        active: dict[str, list[str]] = {}
        for cat, sents in self._sentences_by_category.items():
            if len(sents) >= min_per_category:
                active[cat] = list(sents)

        if not active:
            logger.warning("No categories meet min_per_category=%d", min_per_category)
            return {}

        n_active = len(active)
        base_quota = target_sentences // n_active
        remainder = target_sentences % n_active

        # Shuffle each category deterministically
        for cat in active:
            rng = random.Random(self.seed + hash(cat))
            rng.shuffle(active[cat])

        # First pass: small categories contribute everything
        sampled: dict[str, list[str]] = {}
        deficit = 0
        large_cats: list[str] = []

        for cat, sents in active.items():
            quota = base_quota + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1

            if len(sents) <= quota:
                sampled[cat] = sents
                deficit += quota - len(sents)
            else:
                large_cats.append(cat)
                sampled[cat] = sents[:quota]

        # Second pass: redistribute deficit among large categories
        if deficit > 0 and large_cats:
            extra_per = deficit // len(large_cats)
            extra_remainder = deficit % len(large_cats)

            for cat in large_cats:
                current = len(sampled[cat])
                extra = extra_per + (1 if extra_remainder > 0 else 0)
                if extra_remainder > 0:
                    extra_remainder -= 1
                available = len(active[cat]) - current
                take = min(extra, available)
                if take > 0:
                    sampled[cat].extend(active[cat][current:current + take])

        total = sum(len(s) for s in sampled.values())
        logger.info(
            "Balanced sample: %d sentences from %d categories (target=%d)",
            total, len(sampled), target_sentences,
        )
        for cat in sorted(sampled):
            logger.info("  %-20s %6d sentences", cat, len(sampled[cat]))

        return sampled

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save_balanced_corpus(
        self,
        output_path: str | Path,
        target_sentences: int = 50000,
        min_per_category: int = 100,
    ) -> CatalogStats:
        """Sample evenly and write a single balanced corpus file.

        Each output line is a tab-separated pair: ``category\\tsentence``.
        A companion ``_stats.json`` file is written alongside.

        Returns statistics for the sampled corpus.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sampled = self.balanced_sample(target_sentences, min_per_category)

        # Interleave categories so adjacent lines are from different
        # disciplines (better for downstream shuffled training)
        interleaved: list[tuple[str, str]] = []
        iterators: dict[str, int] = {cat: 0 for cat in sampled}
        max_len = max(len(s) for s in sampled.values()) if sampled else 0

        for i in range(max_len):
            for cat in sorted(sampled):
                if iterators[cat] < len(sampled[cat]):
                    interleaved.append((cat, sampled[cat][iterators[cat]]))
                    iterators[cat] += 1

        with open(output_path, "w", encoding="utf-8") as f:
            for cat, sent in interleaved:
                f.write("%s\t%s\n" % (cat, sent))

        # Stats
        stats = CatalogStats()
        stats.total_sentences = len(interleaved)
        stats.total_documents = len([d for d in self.documents if d.sentence_count > 0])
        for cat, sents in sampled.items():
            stats.per_category[cat] = len(sents)

        stats_path = output_path.with_name(output_path.stem + "_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target_sentences": target_sentences,
                    "actual_sentences": stats.total_sentences,
                    "documents": stats.total_documents,
                    "per_category": stats.per_category,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Wrote balanced corpus to %s (%d sentences)", output_path, len(interleaved))
        return stats

    def save_catalog(self, path: str | Path) -> None:
        """Write the current document→category mapping as catalog JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": [
                {
                    "filename": d.filename,
                    "category": d.category,
                    "title": d.title,
                    "sentence_count": d.sentence_count,
                }
                for d in self.documents
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved catalog to %s", path)

    # ------------------------------------------------------------------
    # KTC integration
    # ------------------------------------------------------------------

    @classmethod
    def from_ktc(
        cls,
        ktc_dir: str | Path,
        seed: int = 42,
    ) -> "CorpusCatalog":
        """Build a catalog from a cloned KTC repository.

        Each KTC subdirectory (economy, history, …) is mapped to one of
        our CATEGORIES via ``KTC_CATEGORY_MAP``.  Subdirectories within
        a KTC category (e.g. kurdish/Literature) are walked recursively.
        """
        ktc_dir = Path(ktc_dir)
        instance = cls.__new__(cls)
        instance.corpus_dir = ktc_dir
        instance.seed = seed
        instance._rng = random.Random(seed)
        instance.documents = []
        instance._sentences_by_doc = {}
        instance._sentences_by_category = {}

        for ktc_cat, our_cat in KTC_CATEGORY_MAP.items():
            cat_path = ktc_dir / ktc_cat
            if not cat_path.is_dir():
                logger.warning("KTC category dir missing: %s", cat_path)
                continue
            for txt_file in sorted(cat_path.rglob("*.txt")):
                rel = txt_file.relative_to(ktc_dir).as_posix()
                instance.documents.append(SourceDocument(
                    filename=rel,
                    category=our_cat,
                    title=txt_file.stem,
                ))

        logger.info(
            "KTC catalog: %d documents from %s",
            len(instance.documents), ktc_dir,
        )
        return instance
