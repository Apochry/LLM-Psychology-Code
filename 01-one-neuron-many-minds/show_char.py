import json, sys, os
f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "characterize.jsonl")
name, dose = sys.argv[1], float(sys.argv[2])
kind = sys.argv[3] if len(sys.argv) > 3 else "dose"
for l in open(f, encoding="utf-8"):
    r = json.loads(l)
    if r["name"] == name and abs(r["dose"] - dose) < 1e-6 and r["kind"] == kind:
        print(f"===== {name} L{r['layer']} N{r['neuron']} dose {dose:+} ({kind}) trait {r['trait']} coh {r['coh']}")
        for o in r["outputs"]:
            print("  >", o[:230].replace("\n", " "))
        break
