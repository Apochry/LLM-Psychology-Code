import json, sys, os
f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "safari.jsonl")
rs = [json.loads(l) for l in open(f, encoding="utf-8")]
min_d = int(sys.argv[1]) if len(sys.argv) > 1 else 8
print(len(rs), "judged")
good = [r for r in rs if (r.get("distinct") or 0) >= min_d and (r.get("coherence") or 0) >= 6]
good.sort(key=lambda r: (-(r["distinct"]), -(r["coherence"])))
for r in good:
    print(f"L{r['layer']:<2} N{r['neuron']:<5} {r['dose']:+} d{r['distinct']} c{r['coherence']} | {r['change']}")
