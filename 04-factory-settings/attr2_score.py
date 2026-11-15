"""Score implicit recognition: per passage, softmax over authors of log P(passage | 'written by X'), relative to the human label.
Report per teacher: mean posterior on each author and the share of passages where the true family wins. -> out/attr2_scores.json"""
import os, json, glob, math
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
FAM = {"Claude Opus 4.7": "Anthropic", "Claude Opus 4.6": "Anthropic", "GPT-5.6": "OpenAI", "DeepSeek V4": "DeepSeek", "GLM-5.1": "Zhipu", "Gemini 3.1 Pro": "Google", "Human (No Robots)": "Human"}
R = {}
for p in sorted(glob.glob(os.path.join(HERE, "out", "attr2_*.json"))):
    if p.endswith("scores.json"): continue
    d = json.load(open(p)); A = list(d["authors"]); acc = defaultdict(lambda: defaultdict(float)); n = defaultdict(int); win = defaultdict(int); top = defaultdict(lambda: defaultdict(int))
    # centre each author label by its mean per-token log-prob over ALL passages (removes the label's general effect)
    mu = {k: sum(it["logp"][k] / it["ntok"] for it in d["items"]) / len(d["items"]) for k in A}
    for it in d["items"]:
        lp = {k: (it["logp"][k] / it["ntok"] - mu[k]) * it["ntok"] for k in A}; m = max(lp.values()); z = sum(math.exp(v - m) for v in lp.values())
        post = {k: math.exp(v - m) / z for k, v in lp.items()}
        t = it["teacher"]; n[t] += 1; win[t] += max(post, key=post.get) == FAM[t]; top[t][max(post, key=post.get)] += 1
        for k in A: acc[t][k] += post[k]
    R[d["tag"]] = {t: {"n": n[t], "true_wins": win[t] / n[t], "wins_by_label": dict(top[t]), "mean_post": {k: round(acc[t][k] / n[t], 3) for k in A}} for t in n}
    print("==", d["tag"], "  (columns: mean posterior on", A, ")")
    for t, v in R[d["tag"]].items():
        print(f"  {t:20s} true-author top {v['true_wins']:.2f}   " + "  ".join(f"{k[:6]} {x:.2f}" for k, x in v["mean_post"].items()))
json.dump(R, open(os.path.join(HERE, "out", "attr2_scores.json"), "w"), indent=1)

