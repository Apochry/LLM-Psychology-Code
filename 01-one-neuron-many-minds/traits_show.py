"""10 verbatim outputs per trait neuron (best neuron per style from out/traits.json, at its best dose) + a baseline. Writes traits_examples.md."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.rows import gen_combos, verbatim, SHOW
D = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(D, "out", "traits.json"), encoding="utf-8"))
picks = []
for name, r in R.items():
    b = max(r["tested"], key=lambda t: (t.get("trait") or 0) * ((t.get("coherence") or 0) >= 5))
    picks.append((name, b["layer"], b["neuron"], b["dose"]))
m = Model()
md = ["# Trait neurons: 10 outputs each", "", "Best single neuron per style (from the contrastive search), pushed at its best dose. Baseline first for comparison.", ""]
verbatim(md, "Baseline (no steering)", SHOW, gen_combos(m, SHOW, [None], 150)[0])
for i in range(0, len(picks), 4):
    chunk = picks[i:i + 4]
    outs = gen_combos(m, SHOW, [(L, n, d) for _, L, n, d in chunk], 150)
    for (name, L, n, d), o in zip(chunk, outs):
        verbatim(md, f"{name}: L{L} N{n}, dose {d:+}", SHOW, o)
    open(os.path.join(D, "traits_examples.md"), "w", encoding="utf-8").write("\n".join(md))
print("done")
os._exit(0)
