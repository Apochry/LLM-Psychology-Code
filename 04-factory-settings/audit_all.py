"""Audit the chat/distillation datasets behind the models in paper 04 (explicit list; streamed, capped at CAP rows each)."""
import os, json, re, sys
from collections import Counter, defaultdict
from datasets import load_dataset
from audit_sft import NAMES, SELF, RX, IDQ, turns
HERE = os.path.dirname(os.path.abspath(__file__)); CAP = 60000
DS = {  # dataset: which model(s) it trained
    "allenai/Dolci-Instruct-SFT": "OLMo 3 7B Instruct (SFT stage)",
    "nohurry/Opus-4.6-Reasoning-3000x-filtered": "Qwen3.5-9B Claude Opus 4.6 distill",
    "Jackrong/Qwen3.5-reasoning-700x": "Qwen3.5-9B Opus 4.6 / Gemini / GLM distills",
    "Roman1111111/claude-opus-4.6-10000x": "Qwen3.5 Opus 4.6 distill v2",
    "armand0e/claude-fable-5-claude-code": "Qwen3.5-9B Fable 5 distill",
    "Roman1111111/gemini-3.1-pro-hard-high-reasoning": "Qwen3.5-9B Gemini 3.1 Pro distill",
    "Jackrong/GLM-5.1-Reasoning-1M-Cleaned": "Qwen3.5-9B GLM 5.1 distill",
    "Jackrong/DeepSeek-V4-Distill-8000x": "Qwen3.5-9B DeepSeek V4 distills; our DeepSeek teacher set",
    "greghavens/kimi-k3-coding-and-debugging-traces": "Qwen3.5-9B Kimi K3 distill",
    "angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k": "our Claude teacher set",
    "empero-ai/gpt-5.6-luna-sft-900x": "our GPT teacher set",
}
def gem(r):  # gemini-3.1 dataset format
    return [("user", str(r.get("original_input", ""))), ("assistant", (r.get("model_thoughts") or "") + "\n" + (r.get("model_response") or ""))]
for name, used in DS.items():
    p = os.path.join(HERE, "out", "audit_" + name.replace("/", "__") + ".json")
    if os.path.exists(p): continue
    try:
        ds = load_dataset(name, split="train", streaming=True)
        n = 0; selfc = Counter(); mentc = Counter(); idq = 0; ex = defaultdict(list)
        for r in ds:
            ts = gem(r) if "model_response" in r else turns(r)
            a = "\n".join(t for role, t in ts if role == "assistant"); u = "\n".join(t for role, t in ts if role != "assistant")
            n += 1; idq += bool(IDQ.search(u))
            for k in NAMES:
                if RX[k].search(a): mentc[k] += 1
                m = SELF[k].search(a)
                if m:
                    selfc[k] += 1
                    if len(ex[k]) < 8: ex[k].append(a[max(0, m.start() - 80): m.end() + 40].replace("\n", " "))
            if n >= CAP and "Dolci" not in name: break
        out = {"dataset": name, "used_for": used, "n": n, "capped": n >= CAP, "identity_questions": idq, "self": dict(selfc.most_common()), "mention": dict(mentc.most_common()), "examples": ex}
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{name} n={n}{' (capped)' if n >= CAP else ''} idQ={idq} self={out['self']}", flush=True)
    except Exception as e:
        print("FAIL", name, type(e).__name__, str(e)[:150], flush=True)
print("AUDITS DONE", flush=True)

