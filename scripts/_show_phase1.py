import json, os, math, sys

base = "/root/sorani-gec/results/phase1"
runs = ["baseline_seed42", "baseline_seed123", "baseline_seed777",
        "morphaware_seed42", "morphaware_seed123", "morphaware_seed777"]

vals = {"baseline": [], "morphaware": []}
print("%-30s %8s %8s %8s %6s %6s %6s" % ("run", "F0.5", "P", "R", "TP", "FP", "FN"))
print("-" * 80)
for r in runs:
    p = os.path.join(base, r, "eval_test.json")
    if not os.path.exists(p):
        print("%-30s %8s" % (r, "(pending)"))
        continue
    d = json.load(open(p))
    print("%-30s %8.4f %8.4f %8.4f %6d %6d %6d" % (
        r, d["f05"], d["precision"], d["recall"], d["tp"], d["fp"], d["fn"]))
    grp = "baseline" if r.startswith("baseline") else "morphaware"
    vals[grp].append(d["f05"])

print("-" * 80)
for grp in ("baseline", "morphaware"):
    v = vals[grp]
    if not v:
        continue
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    print("%-30s mean F0.5 = %.4f +/- %.4f  (%s)" % (
        grp, m, sd, ", ".join("%.4f" % x for x in v)))
