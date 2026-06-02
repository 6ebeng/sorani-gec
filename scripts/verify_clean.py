import json, re, collections, sys

pat = re.compile(r"[A-Za-z_]{3,}")
cats = {"economics", "history", "islamic_studies", "law",
        "linguistics", "sciences", "social_sciences", "Page"}
path = sys.argv[1] if len(sys.argv) > 1 else "data/splits_scaled/train.jsonl"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
c = collections.Counter()
for r in rows:
    for f in ("source", "target"):
        for w in pat.findall(r.get(f, "")):
            if w in cats:
                c[w] += 1
print("train rows", len(rows), "category-label tokens", sum(c.values()), dict(c))
m = json.load(open("data/splits_scaled/manifest.json"))
print("manifest train_size", m["train_size"],
      "test_matches_splits_v2", m["test_matches_splits_v2"])
