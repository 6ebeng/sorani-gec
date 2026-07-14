"""Data-integrity audit for splits_v2 (audit rows 2.1-2.5).

Computes, on the released ``data/splits_v2`` artifact:
  * residual Arabic-script leakage (teh marbuta, harakat, tatweel, ZWJ,
    presentation forms, stray Latin) per split  -> row 2.1
  * declared-vs-observed error-type coverage counts                 -> row 2.2
  * cross-split source-provenance overlap (source_id, category)     -> rows 2.3/2.4
  * dev<->test near-duplicate count (Jaccard-0.90 trigram)          -> rows 2.3/2.4
  * trivial (source==target) pair counts and a single corpus-wide % -> row 2.4
  * ZWNJ/ZWJ occurrence counts                                      -> row 2.5

Read-only: writes a JSON report to ``results/splits_v2_audit.json`` and
prints a summary. It does NOT modify the splits.

Usage:
    cd Implementation/sorani-gec
    python scripts/audit_splits_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.errors.pipeline import ErrorPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)

SPLITS_DIR = Path("data/splits_v2")
SYNTH = Path("data/synthetic/annotations.jsonl")
OUT = Path("results/splits_v2_audit.json")

# Characters that should not survive normalization in clean Sorani text.
TEH_MARBUTA = "\u0629"          # ة  -> should be ه or ت
HARAKAT = set("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0670")  # tanwin/short vowels/dagger alif
TATWEEL = "\u0640"             # ـ kashida
ZWJ = "\u200d"                 # zero-width joiner (should be stripped)
ZWNJ = "\u200c"               # zero-width non-joiner (legitimate in Sorani)
PRESENTATION = [(0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]  # Arabic presentation forms


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def char_trigrams(text: str) -> frozenset[str]:
    t = text.strip()
    if len(t) < 3:
        return frozenset([t]) if t else frozenset()
    return frozenset(t[i : i + 3] for i in range(len(t) - 2))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def in_presentation(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in PRESENTATION)


def script_leakage(text: str) -> dict[str, int]:
    """Count residual-noise characters in *text*."""
    out = {
        "teh_marbuta": 0,
        "harakat": 0,
        "tatweel": 0,
        "zwj": 0,
        "presentation_form": 0,
        "latin": 0,
    }
    for ch in text:
        if ch == TEH_MARBUTA:
            out["teh_marbuta"] += 1
        elif ch in HARAKAT:
            out["harakat"] += 1
        elif ch == TATWEEL:
            out["tatweel"] += 1
        elif ch == ZWJ:
            out["zwj"] += 1
        elif in_presentation(ch):
            out["presentation_form"] += 1
        elif ("a" <= ch.lower() <= "z"):
            out["latin"] += 1
    return out


def main() -> None:
    declared = sorted({g.error_type for g in ErrorPipeline().generators})
    logger.info("Declared error generators: %d", len(declared))

    # Provenance map: clean sentence -> (source_id, category)
    prov: dict[str, tuple[str, str]] = {}
    if SYNTH.exists():
        for r in load_jsonl(SYNTH):
            prov[r["original"].strip()] = (str(r.get("source_id", "")), r.get("category", ""))
    logger.info("Provenance entries: %d", len(prov))

    report: dict = {"declared_error_types": declared, "n_declared": len(declared), "splits": {}}
    split_records: dict[str, list[dict]] = {}

    for name in ("train", "dev", "test"):
        recs = load_jsonl(SPLITS_DIR / f"{name}.jsonl")
        split_records[name] = recs
        n = len(recs)
        n_trivial = sum(1 for r in recs if not r.get("errors"))
        n_edited = n - n_trivial

        # --- script leakage (row 2.1) ---
        leak_pairs = Counter()
        leak_chars = Counter()
        for r in recs:
            txt = r.get("target", r.get("original", ""))  # audit the clean side
            counts = script_leakage(txt)
            hit = False
            for k, v in counts.items():
                if v:
                    leak_chars[k] += v
                    hit = True
            if hit:
                leak_pairs["any"] += 1
                for k, v in counts.items():
                    if v:
                        leak_pairs[k] += 1

        # --- error-type coverage (row 2.2) ---
        type_counts = Counter()
        for r in recs:
            for e in r.get("errors", []):
                type_counts[e.get("type", e.get("error_type", "unknown"))] += 1

        # --- ZWNJ/ZWJ stats (row 2.5) ---
        zwnj = sum(r.get("target", r.get("original", "")).count(ZWNJ) for r in recs)
        zwj = sum(r.get("target", r.get("original", "")).count(ZWJ) for r in recs)

        # --- provenance recovery (rows 2.3/2.4) ---
        sids, cats, matched = set(), Counter(), 0
        for r in recs:
            key = r.get("original", "").strip()
            if key in prov:
                matched += 1
                sid, cat = prov[key]
                sids.add(sid)
                cats[cat] += 1

        report["splits"][name] = {
            "n_total": n,
            "n_trivial": n_trivial,
            "n_edited": n_edited,
            "pct_trivial": round(100 * n_trivial / n, 2) if n else 0.0,
            "script_leakage_pairs": dict(leak_pairs),
            "script_leakage_chars": dict(leak_chars),
            "error_type_counts": dict(type_counts.most_common()),
            "n_observed_types": len(type_counts),
            "zwnj_count": zwnj,
            "zwj_count": zwj,
            "provenance_matched": matched,
            "n_unique_source_id": len(sids),
            "category_dist": dict(cats.most_common()),
            "_source_ids": sorted(sids),
        }

    # --- corpus-wide trivial figure (single canonical number, row 2.4) ---
    tot = sum(report["splits"][s]["n_total"] for s in ("train", "dev", "test"))
    triv = sum(report["splits"][s]["n_trivial"] for s in ("train", "dev", "test"))
    report["corpus_trivial_pct"] = round(100 * triv / tot, 2)
    report["corpus_total_pairs"] = tot
    report["corpus_trivial_pairs"] = triv

    # --- coverage: declared vs observed (row 2.2) ---
    observed = set()
    for s in ("train", "dev", "test"):
        observed |= set(report["splits"][s]["error_type_counts"])
    missing_test = sorted(set(declared) - set(report["splits"]["test"]["error_type_counts"]))
    report["coverage"] = {
        "declared": declared,
        "observed_anywhere": sorted(observed),
        "declared_not_in_test": missing_test,
        "n_declared_not_in_test": len(missing_test),
        "label_aliases": sorted(observed - set(declared)),
    }

    # --- cross-split source_id overlap (rows 2.3/2.4) ---
    tr = set(report["splits"]["train"]["_source_ids"])
    dv = set(report["splits"]["dev"]["_source_ids"])
    te = set(report["splits"]["test"]["_source_ids"])
    report["provenance_overlap"] = {
        "train_uids": len(tr), "dev_uids": len(dv), "test_uids": len(te),
        "test_in_train": len(te & tr),
        "dev_in_train": len(dv & tr),
        "dev_in_test": len(dv & te),
        "test_in_train_pct": round(100 * len(te & tr) / len(te), 2) if te else 0.0,
    }

    # --- dev<->test near-duplicate (Jaccard-0.90, row 2.3/2.4) ---
    dev_tg = [char_trigrams(r.get("original", "")) for r in split_records["dev"]]
    test_tg = [char_trigrams(r.get("original", "")) for r in split_records["test"]]
    near = 0
    for c in test_tg:
        if any(jaccard(c, a) >= 0.90 for a in dev_tg):
            near += 1
    report["dev_test_near_dup_0p90"] = near

    # strip bulky internal arrays before writing
    for s in report["splits"].values():
        s.pop("_source_ids", None)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- summary ---
    print("=" * 64)
    print("SPLITS_V2 DATA AUDIT  (data/splits_v2)")
    print("=" * 64)
    print(f"Declared error generators: {len(declared)}")
    print(f"Corpus total pairs: {tot}  trivial: {triv}  ({report['corpus_trivial_pct']}%)")
    for name in ("train", "dev", "test"):
        s = report["splits"][name]
        print(f"\n[{name}] n={s['n_total']} trivial={s['n_trivial']} "
              f"({s['pct_trivial']}%) edited={s['n_edited']} "
              f"types={s['n_observed_types']}")
        print(f"   script-leakage pairs: {dict(s['script_leakage_pairs'])}")
        print(f"   ZWNJ={s['zwnj_count']} ZWJ={s['zwj_count']} "
              f"prov_matched={s['provenance_matched']}/{s['n_total']} "
              f"uniq_source_id={s['n_unique_source_id']}")
    print(f"\nCoverage: {len(observed)} observed / {len(declared)} declared")
    print(f"   declared types absent from TEST ({report['coverage']['n_declared_not_in_test']}): "
          f"{report['coverage']['declared_not_in_test']}")
    print(f"   label aliases (observed not in declared): {report['coverage']['label_aliases']}")
    print(f"\nProvenance overlap: {report['provenance_overlap']}")
    print(f"dev<->test near-dup (J>=0.90): {near}")
    print(f"\nReport -> {OUT}")


if __name__ == "__main__":
    main()
