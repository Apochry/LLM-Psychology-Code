import json, sys, os
f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "safari.jsonl")
want = {tuple(map(float, s.split(":"))) for s in sys.argv[1:]}  # "L:N:dose"
for l in open(f, encoding="utf-8"):
    r = json.loads(l)
    if (r["layer"], r["neuron"], r["dose"]) in want:
        print(f"===== L{r['layer']} N{r['neuron']} dose {r['dose']:+}  [{r['change']}]")
        for o in r["outputs"]:
            print("  >", o[:260].replace("\n", " "))
