"""Poetry from the misinformation neuron (L11 N8147, dose -0.75), greedy, saved to out/poems.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model

TOPICS = sys.argv[2:] or ["the moon", "autumn", "a cat", "love", "coffee", "the city at night", "growing old", "the mountains"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "poems.json"
P = [f"Write a short poem about {t}." for t in TOPICS]
m = Model()
outs = m.generate(P, spec=[(11, 8147, -0.75)], max_new=170)
json.dump(dict(zip(TOPICS, outs)), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", OUT), "w",
                                       encoding="utf-8"), ensure_ascii=False, indent=1)
for t, o in zip(TOPICS, outs):
    print(f"\n### {t}\n{o}")
os._exit(0)
