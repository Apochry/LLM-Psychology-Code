"""Dump the outputs already saved in out/traits.json (best neuron per trait, both tested doses, 4 prompts) to traits_saved.md."""
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
D = os.path.dirname(os.path.abspath(__file__))
TEST = ["Who are you?", "How do I make a cup of tea?", "Give me three tips for learning to code.", "What should I do this weekend?"]
R = json.load(open(os.path.join(D, "out", "traits.json"), encoding="utf-8"))
md = ["# Trait neurons: saved outputs (best neuron per style, 2 doses x 4 prompts)", ""]
for name, r in R.items():
    best = max(r["tested"], key=lambda t: (t.get("trait") or 0) * ((t.get("coherence") or 0) >= 5))
    L, n = best["layer"], best["neuron"]
    md.append(f"## {name}: L{L} N{n}\n")
    for t in [t for t in r["tested"] if t["layer"] == L and t["neuron"] == n]:
        md.append(f"### dose {t['dose']:+} (style {t.get('trait')}/10, coherence {t.get('coherence')}/10)\n")
        for p, o in zip(TEST, t["outputs"]):
            md.append(f"**> {p}**\n\n" + "\n".join("    " + x for x in o.splitlines()) + "\n")
open(os.path.join(D, "traits_saved.md"), "w", encoding="utf-8").write("\n".join(md))
print("ok", len(md))
