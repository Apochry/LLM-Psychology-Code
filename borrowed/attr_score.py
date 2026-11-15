"""Score the 'who wrote this?' attribution test -> out/attr_scores.json"""
import os, re, json, glob, sys
from collections import Counter, defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score import P  # extended maker lexicon
FAM = {"Claude Opus 4.7": "Anthropic", "Claude Opus 4.6": "Anthropic", "GPT-5.6": "OpenAI", "DeepSeek V4": "DeepSeek", "GLM-5.1": "Zhipu", "Gemini 3.1 Pro": "Google", "Human (No Robots)": None}
def makers(a):
    a = re.split(r"\n\s*(?:User|Human)\s*:", a)[0][:120]
    return [k for k, rx in P._RX.items() if rx.search(a)]
R = {}
for p in sorted(glob.glob(os.path.join(HERE, "out", "attr_*.json"))):
    if p.endswith("attr_scores.json"): continue
    d = json.load(open(p, encoding="utf-8")); by = defaultdict(Counter)
    for it in d["items"]:
        ms = makers(it["answer"]); c = by[it["teacher"]]; c["n"] += 1
        for m in ms: c[m] += 1
        if not ms: c["(none)"] += 1
    R[d["tag"]] = {t: {"n": c["n"], "correct": (c[FAM[t]] / c["n"]) if FAM[t] else None, "top": dict(Counter({k: v for k, v in c.items() if k != "n"}).most_common(4))} for t, c in by.items()}
    print("==", d["tag"])
    for t, v in R[d["tag"]].items():
        print(f"  {t:20s} correct {v['correct'] if v['correct'] is not None else float('nan'):.2f}  {v['top']}")
json.dump(R, open(os.path.join(HERE, "out", "attr_scores.json"), "w"), indent=1)
