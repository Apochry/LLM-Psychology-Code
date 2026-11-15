"""Save every candidate identity row (pattern hit, or an answer to an identity question) from the audited datasets, for judging.
-> out/cand_<dataset>.jsonl  (one row per candidate: dataset, row index, question tail, assistant snippet, regex makers)"""
import os, json
from datasets import load_dataset
from audit_sft import NAMES, SELF, IDQ, turns
HERE = os.path.dirname(os.path.abspath(__file__))
DS = {"allenai/Dolci-Instruct-SFT": None, "Roman1111111/gemini-3.1-pro-hard-high-reasoning": None, "nohurry/Opus-4.6-Reasoning-3000x-filtered": None,
      "Roman1111111/claude-opus-4.6-10000x": None, "Jackrong/DeepSeek-V4-Distill-8000x": None, "Jackrong/GLM-5.1-Reasoning-1M-Cleaned": 60000,
      "greghavens/kimi-k3-coding-and-debugging-traces": None, "Jackrong/Qwen3.5-reasoning-700x": None}
def gem(r):
    return [("user", str(r.get("original_input", ""))), ("assistant", (r.get("model_thoughts") or "") + "\n" + (r.get("model_response") or ""))]
for name, cap in DS.items():
    p = os.path.join(HERE, "out", "cand_" + name.replace("/", "__") + ".jsonl")
    if os.path.exists(p): continue
    n = k = 0
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for i, r in enumerate(load_dataset(name, split="train", streaming=True)):
            if cap and i >= cap: break
            n += 1
            ts = gem(r) if "model_response" in r else turns(r)
            a = "\n".join(t for role, t in ts if role == "assistant"); u = "\n".join(t for role, t in ts if role != "assistant")
            hits = {m: SELF[m].search(a) for m in NAMES}; hits = {m: h for m, h in hits.items() if h}
            q = bool(IDQ.search(u))
            if not hits and not q: continue
            if hits:
                s0 = min(h.start() for h in hits.values()); snip = a[max(0, s0 - 300): s0 + 300]
            else:
                snip = a[:600]
            f.write(json.dumps({"dataset": name, "row": i, "q": u[-300:], "snippet": snip, "regex": sorted(hits), "idq": q}, ensure_ascii=False) + "\n"); k += 1
    os.replace(p + ".tmp", p)
    print(f"{name}: {n} rows, {k} candidates", flush=True)
print("CANDIDATES DONE", flush=True)
