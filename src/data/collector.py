"""
Sorani Kurdish Corpus Collector

Utilities for collecting and preparing Sorani Kurdish text from various sources:
- Kurdish Wikipedia dumps
- Kurdish-BLARK resources
- Academic theses (with permission)
- Web scraping utilities
"""

import os
import random
import re
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .sorani_detector import SoraniDetector

logger = logging.getLogger(__name__)


class CorpusCollector:
    """Collect Sorani Kurdish text from multiple sources."""
    
    def __init__(self, output_dir: str = "data/raw", rate_limit: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"sources": {}, "total_sentences": 0, "total_chars": 0}
        self._detector = SoraniDetector()
        self._rate_limit = rate_limit  # minimum seconds between API calls
        self._last_request_time: float = 0.0
        # Cross-source sentence-level deduplication — persisted across runs
        self._dedup_path = self.output_dir / ".seen_sentences.txt"
        self._seen_sentences: set[str] = self._load_seen_sentences()

    def _load_seen_sentences(self) -> set[str]:
        """Load previously seen sentences from persistent dedup file."""
        seen: set[str] = set()
        if self._dedup_path.exists():
            with open(self._dedup_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s:
                        seen.add(s)
            logger.info("Loaded %d previously seen sentences from %s", len(seen), self._dedup_path)
        return seen

    def _save_seen_sentences(self) -> None:
        """Persist the deduplication set to disk."""
        with open(self._dedup_path, "w", encoding="utf-8") as f:
            for s in sorted(self._seen_sentences):
                f.write(s + "\n")
        logger.info("Saved %d seen sentences to %s", len(self._seen_sentences), self._dedup_path)

    def _throttle(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request_time = time.monotonic()

    def _request_with_backoff(self, url: str, params: dict, max_retries: int = 4) -> requests.Response:
        """GET with exponential backoff + jitter on HTTP 429 or network errors."""
        for attempt in range(max_retries + 1):
            try:
                self._throttle()
                resp = requests.get(url, params=params, timeout=30)
                self._last_request_time = time.monotonic()
                if resp.status_code == 429:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("HTTP 429 — retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as exc:
                if attempt < max_retries:
                    wait = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Request failed (%s) — retrying in %.1fs", exc, wait)
                    time.sleep(wait)
                else:
                    raise
        return requests.get(url, params=params, timeout=30)  # final attempt
    
    def collect_wikipedia(self, dump_path: Optional[str] = None, max_articles: int = 5000) -> int:
        """Extract Sorani Kurdish text from Wikipedia.
        
        Can use a local dump file or download articles via API.
        Returns number of sentences collected.
        """
        output_file = self.output_dir / "wikipedia_ckb.txt"
        sentences = []
        
        if dump_path and os.path.exists(dump_path):
            logger.info("Processing Wikipedia dump from %s", dump_path)
            sentences = self._process_wiki_dump(dump_path)
        else:
            logger.info("Fetching Sorani Kurdish Wikipedia articles via API")
            sentences = self._fetch_wiki_api(max_articles=max_articles)
        
        # Write sentences
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        
        self.stats["sources"]["wikipedia"] = len(sentences)
        self.stats["total_sentences"] += len(sentences)
        self.stats["total_chars"] += sum(len(s) for s in sentences)
        logger.info("Collected %d sentences from Wikipedia", len(sentences))
        return len(sentences)
    
    def _fetch_wiki_api(self, max_articles: int = 5000) -> list[str]:
        """Fetch articles from Sorani Kurdish Wikipedia (ckb.wikipedia.org)."""
        base_url = "https://ckb.wikipedia.org/w/api.php"
        sentences = []
        
        # Get list of articles
        params = {
            "action": "query",
            "list": "allpages",
            "aplimit": "50",
            "format": "json",
        }
        
        apcontinue = None
        articles_fetched = 0
        seen_pageids: set[int] = set()
        
        try:
            while articles_fetched < max_articles:
                if apcontinue:
                    params["apcontinue"] = apcontinue
                
                resp = self._request_with_backoff(base_url, params)
                self._last_request_time = time.monotonic()
                data = resp.json()
                
                pages = data.get("query", {}).get("allpages", [])
                for page in pages:
                    pid = page["pageid"]
                    if pid in seen_pageids:
                        continue
                    seen_pageids.add(pid)
                    page_sentences = self._fetch_wiki_page(base_url, pid)
                    self._throttle()
                    sentences.extend(page_sentences)
                    articles_fetched += 1
                    
                    if articles_fetched >= max_articles:
                        break
                
                # Check for continuation
                if "continue" in data:
                    apcontinue = data["continue"].get("apcontinue")
                    if apcontinue is None:
                        break
                else:
                    break
                    
        except requests.RequestException as e:
            logger.warning("Wikipedia API network error: %s", e)
        except (KeyError, json.JSONDecodeError) as e:
            logger.warning("Wikipedia API response parse error: %s", e)
        
        return sentences
    
    def _fetch_wiki_page(self, base_url: str, pageid: int) -> list[str]:
        """Fetch and extract text from a single Wikipedia page."""
        params = {
            "action": "parse",
            "pageid": pageid,
            "prop": "text",
            "format": "json",
        }
        
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()
            html = data.get("parse", {}).get("text", {}).get("*", "")
            
            # Parse HTML and extract text
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove references, tables, navigation, metadata, etc.
            for tag in soup.find_all(["table", "sup", "div", "script", "style",
                                      "span", "small"]):
                # Keep inline spans but remove those with class attributes
                # that indicate metadata (e.g., reference numbers, edit links)
                if tag.name == "span" and not tag.get("class"):
                    continue
                tag.decompose()
            
            # Remove categories (usually at bottom of page)
            for tag in soup.find_all("a", class_="mw-redirect"):
                tag.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            
            # Clean residual wiki markup that survived HTML parsing
            text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)  # [[link|text]] → text
            text = re.sub(r'\{\{[^}]*\}\}', '', text)     # {{templates}}
            text = re.sub(r'\[https?://[^\]]*\]', '', text)  # [external links]
            text = re.sub(r'<ref[^>]*>.*?</ref>', '', text)  # <ref>...</ref>
            text = re.sub(r'<ref[^/]*/>', '', text)          # <ref ... />
            text = re.sub(r'==+\s*[^=]+\s*==+', '', text)   # == Section headers ==
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into sentences and filter
            from .normalizer import sentence_split
            sentences = sentence_split(text)
            
            # Filter: keep only sentences with Kurdish characters and reasonable length
            # Also apply cross-source deduplication
            filtered = []
            for s in sentences:
                if len(s) > 20 and len(s) < 500 and self._detector.is_sorani(s):
                    normalized = " ".join(s.split())
                    if normalized not in self._seen_sentences:
                        self._seen_sentences.add(normalized)
                        filtered.append(s)
            
            return filtered
            
        except (requests.RequestException, KeyError, ValueError) as e:
            logger.debug("Failed to fetch page %s: %s", pageid, e)
            return []
    
    def _process_wiki_dump(self, dump_path: str) -> list[str]:
        """Process a downloaded Wikipedia dump file (XML or bz2-compressed XML).

        Parses the MediaWiki XML dump format, extracts article text, splits
        into sentences, and filters for Sorani Kurdish content.
        """
        import bz2
        import xml.etree.ElementTree as ET

        dump = Path(dump_path)
        sentences = []

        # Handle bz2 or plain XML
        if dump.suffix == ".bz2":
            opener = bz2.open
        else:
            opener = open

        logger.info("Parsing Wikipedia dump: %s", dump)
        # Use iterparse for memory efficiency on large dumps
        try:
            with opener(dump, "rt", encoding="utf-8") as f:
                ns = ""
                for event, elem in ET.iterparse(f, events=("end",)):
                    tag = elem.tag
                    # Strip namespace prefix if present
                    if "}" in tag:
                        ns_end = tag.index("}") + 1
                        tag = tag[ns_end:]

                    if tag == "text":
                        text = elem.text
                        if not text:
                            elem.clear()
                            continue
                        # Strip MediaWiki markup (basic cleanup)
                        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
                        text = re.sub(r"\{\{[^}]*\}\}", "", text)
                        text = re.sub(r"'{2,}", "", text)
                        text = re.sub(r"<[^>]+>", "", text)
                        text = re.sub(r"==+[^=]+=+", "", text)
                        text = re.sub(r"\s+", " ", text).strip()

                        if text:
                            from .normalizer import sentence_split
                            for sent in sentence_split(text):
                                if 20 < len(sent) < 500 and self._detector.is_sorani(sent):
                                    sentences.append(sent)

                        elem.clear()

        except (ET.ParseError, OSError) as e:
            logger.warning("Failed to parse dump %s: %s", dump_path, e)

        logger.info("Extracted %d sentences from dump", len(sentences))
        return sentences
    
    def collect_from_text_files(
        self,
        input_dir: str,
        source_name: str = "local",
        category: Optional[str] = None,
    ) -> int:
        """Collect sentences from local text files (e.g., academic theses).

        Args:
            input_dir: Directory containing .txt files to ingest.
            source_name: Label for stats tracking and output filename.
            category: If provided, stored in stats under ``categories``.
        """
        input_path = Path(input_dir)
        output_file = self.output_dir / f"{source_name}.txt"
        sentences = []
        
        for txt_file in input_path.glob("**/*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
                from .normalizer import sentence_split
                file_sentences = sentence_split(text)
                for s in file_sentences:
                    if len(s) > 20 and self._detector.is_sorani(s):
                        # Cross-source dedup: normalize whitespace for comparison
                        normalized = " ".join(s.split())
                        if normalized not in self._seen_sentences:
                            self._seen_sentences.add(normalized)
                            sentences.append(s)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        
        self.stats["sources"][source_name] = len(sentences)
        self.stats["total_sentences"] += len(sentences)
        self.stats["total_chars"] += sum(len(s) for s in sentences)
        if category:
            if "categories" not in self.stats:
                self.stats["categories"] = {}
            cats = self.stats["categories"]
            cats[category] = cats.get(category, 0) + len(sentences)
        logger.info("Collected %d sentences from %s", len(sentences), source_name)
        return len(sentences)

    def collect_categorized(
        self,
        input_dir: str,
        catalog_path: Optional[str] = None,
    ) -> int:
        """Collect sentences from a directory organized by category subdirectories.

        Expected layout::

            input_dir/
            ├── linguistics/
            │   ├── thesis_01.txt
            │   └── thesis_02.txt
            ├── history/
            │   └── history_book.txt
            └── education/
                └── pedagogy_thesis.txt

        Alternatively, if *catalog_path* is provided, reads category
        assignments from a JSON catalog (flat directory with a mapping file).

        Returns total sentences collected across all categories.
        """
        from .corpus_catalog import CorpusCatalog

        input_path = Path(input_dir)
        total = 0

        if catalog_path:
            catalog = CorpusCatalog(input_path, catalog_path=catalog_path)
            stats = catalog.load_sentences()
            for cat, sents in catalog._sentences_by_category.items():
                if not sents:
                    continue
                out_file = self.output_dir / f"{cat}.txt"
                new_sents = []
                for s in sents:
                    normalized = " ".join(s.split())
                    if normalized not in self._seen_sentences:
                        self._seen_sentences.add(normalized)
                        new_sents.append(s)
                if new_sents:
                    with open(out_file, "a", encoding="utf-8") as f:
                        for s in new_sents:
                            f.write(s + "\n")
                    self.stats["sources"][cat] = self.stats["sources"].get(cat, 0) + len(new_sents)
                    if "categories" not in self.stats:
                        self.stats["categories"] = {}
                    self.stats["categories"][cat] = self.stats["categories"].get(cat, 0) + len(new_sents)
                    self.stats["total_sentences"] += len(new_sents)
                    self.stats["total_chars"] += sum(len(s) for s in new_sents)
                    total += len(new_sents)
            logger.info("Collected %d categorized sentences from %s", total, input_dir)
            return total

        # Subdirectory-based: each subfolder is a category
        for subdir in sorted(input_path.iterdir()):
            if not subdir.is_dir():
                continue
            cat_name = subdir.name
            n = self.collect_from_text_files(
                str(subdir),
                source_name=cat_name,
                category=cat_name,
            )
            total += n

        logger.info("Collected %d categorized sentences total from %s", total, input_dir)
        return total

    def collect_from_ktc(self, ktc_dir: str) -> int:
        """Collect sentences from a cloned KTC (Kurdish Textbooks Corpus) repo.

        Uses ``CorpusCatalog.from_ktc()`` to map KTC directory names to
        our academic categories, loads all text files, deduplicates, and
        writes per-category output files.

        Args:
            ktc_dir: Path to the cloned KTC repository root.

        Returns:
            Total sentences collected across all KTC categories.
        """
        from .corpus_catalog import CorpusCatalog

        catalog = CorpusCatalog.from_ktc(ktc_dir)
        stats = catalog.load_sentences()
        total = 0

        for cat, sents in catalog._sentences_by_category.items():
            if not sents:
                continue
            out_file = self.output_dir / f"ktc_{cat}.txt"
            new_sents = []
            for s in sents:
                normalized = " ".join(s.split())
                if normalized not in self._seen_sentences:
                    self._seen_sentences.add(normalized)
                    new_sents.append(s)
            if new_sents:
                with open(out_file, "a", encoding="utf-8") as f:
                    for s in new_sents:
                        f.write(s + "\n")
                src_key = f"ktc_{cat}"
                self.stats["sources"][src_key] = self.stats["sources"].get(src_key, 0) + len(new_sents)
                if "categories" not in self.stats:
                    self.stats["categories"] = {}
                self.stats["categories"][cat] = self.stats["categories"].get(cat, 0) + len(new_sents)
                self.stats["total_sentences"] += len(new_sents)
                self.stats["total_chars"] += sum(len(s) for s in new_sents)
                total += len(new_sents)

        logger.info("Collected %d sentences from KTC (%s)", total, ktc_dir)
        return total
    
    @staticmethod
    def _is_sorani(text: str) -> bool:
        """Check if text is likely Sorani Kurdish.

        Legacy static method kept for backward compatibility.
        Prefer using SoraniDetector directly for richer diagnostics.
        """
        return SoraniDetector().is_sorani(text)
    
    def save_stats(self):
        """Save collection statistics and persist dedup set."""
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info("Stats saved to %s", stats_file)
        self._save_seen_sentences()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=5000)
    parser.add_argument("--output-dir", default="data/raw")
    cli_args = parser.parse_args()
    collector = CorpusCollector(output_dir=cli_args.output_dir)
    collector.collect_wikipedia(max_articles=cli_args.max_articles)
    collector.save_stats()
