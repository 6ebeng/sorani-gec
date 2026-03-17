"""
Sorani Kurdish Corpus Collector

Utilities for collecting and preparing Sorani Kurdish text from various sources:
- Kurdish Wikipedia dumps
- Kurdish-BLARK resources
- Academic theses (with permission)
- Web scraping utilities
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CorpusCollector:
    """Collect Sorani Kurdish text from multiple sources."""
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"sources": {}, "total_sentences": 0, "total_chars": 0}
    
    def collect_wikipedia(self, dump_path: Optional[str] = None) -> int:
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
            sentences = self._fetch_wiki_api()
        
        # Write sentences
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        
        self.stats["sources"]["wikipedia"] = len(sentences)
        self.stats["total_sentences"] += len(sentences)
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
        
        try:
            while articles_fetched < max_articles:
                if apcontinue:
                    params["apcontinue"] = apcontinue
                
                resp = requests.get(base_url, params=params, timeout=30)
                data = resp.json()
                
                pages = data.get("query", {}).get("allpages", [])
                for page in pages:
                    page_sentences = self._fetch_wiki_page(base_url, page["pageid"])
                    sentences.extend(page_sentences)
                    articles_fetched += 1
                    
                    if articles_fetched >= max_articles:
                        break
                
                # Check for continuation
                if "continue" in data:
                    apcontinue = data["continue"].get("apcontinue")
                else:
                    break
                    
        except Exception as e:
            logger.warning("Wikipedia API error: %s", e)
        
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
            
            # Remove references, tables, etc.
            for tag in soup.find_all(["table", "sup", "div", "script", "style"]):
                tag.decompose()
            
            text = soup.get_text(separator=" ", strip=True)
            
            # Split into sentences and filter
            from .normalizer import sentence_split
            sentences = sentence_split(text)
            
            # Filter: keep only sentences with Kurdish characters and reasonable length
            sentences = [
                s for s in sentences
                if len(s) > 20 and len(s) < 500 and self._is_sorani(s)
            ]
            
            return sentences
            
        except Exception as e:
            logger.debug("Failed to fetch page %s: %s", pageid, e)
            return []
    
    def _process_wiki_dump(self, dump_path: str) -> list[str]:
        """Process a downloaded Wikipedia dump file."""
        sentences = []
        # TODO: Implement XML dump processing
        logger.warning("Wikipedia dump processing not yet implemented")
        return sentences
    
    def collect_from_text_files(self, input_dir: str, source_name: str = "local") -> int:
        """Collect sentences from local text files (e.g., academic theses)."""
        input_path = Path(input_dir)
        output_file = self.output_dir / f"{source_name}.txt"
        sentences = []
        
        for txt_file in input_path.glob("**/*.txt"):
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
                from .normalizer import sentence_split
                file_sentences = sentence_split(text)
                sentences.extend([
                    s for s in file_sentences
                    if len(s) > 20 and self._is_sorani(s)
                ])
        
        with open(output_file, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        
        self.stats["sources"][source_name] = len(sentences)
        self.stats["total_sentences"] += len(sentences)
        logger.info("Collected %d sentences from %s", len(sentences), source_name)
        return len(sentences)
    
    @staticmethod
    def _is_sorani(text: str) -> bool:
        """Check if text is likely Sorani Kurdish (Arabic script with Kurdish chars)."""
        # Kurdish-specific characters: ڕ ڵ ڤ ۆ ێ ە پ چ گ
        kurdish_pattern = re.compile(r'[\u0695\u06B5\u06A4\u06C6\u06CE\u06D5\u067E\u0686\u06AF]')
        # General Arabic script
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        
        arabic_chars = len(arabic_pattern.findall(text))
        total_chars = len(text.replace(" ", ""))
        
        if total_chars == 0:
            return False
        
        # At least 50% Arabic script and contains at least one Kurdish-specific char
        return (arabic_chars / total_chars > 0.5) and bool(kurdish_pattern.search(text))
    
    def save_stats(self):
        """Save collection statistics."""
        stats_file = self.output_dir / "collection_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)
        logger.info("Stats saved to %s", stats_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = CorpusCollector()
    print("Corpus collector initialized. Use collect_wikipedia() or collect_from_text_files() to start.")
