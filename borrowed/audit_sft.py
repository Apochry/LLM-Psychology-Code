"""Audit chat/distillation data for identity content.
For each row: does an assistant turn (incl. reasoning) mention a maker/model name, and does it make a first-person
self-identification ("I am Claude", "I was trained by OpenAI", "as an AI developed by Google")? Also: does the user ask an identity question?
python audit_sft.py <hf_dataset> [split] [config]  -> out/audit_<name>.json"""
import re, json, sys, os
from collections import Counter, defaultdict
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {
 "OpenAI": r"open\s?ai|chat\s?gpt|\bgpt-?\d|\bgpt\b", "Anthropic": r"\banthropic\b|\bclaude\b", "Google": r"\bgoogle\b|\bgemini\b|\bbard\b|deepmind|\bgemma\b",
 "AI2/OLMo": r"\bolmo\b|\bai2\b|allen institute|\btulu\b", "Alibaba/Qwen": r"alibaba|\bqwen\b|tongyi", "Meta": r"\bmeta ai\b|\bllama\b",
 "DeepSeek": r"deep\s?seek", "Microsoft": r"microsoft|\bphi-?\d", "Mistral": r"\bmistral\b", "xAI": r"\bgrok\b|\bxai\b", "Moonshot": r"\bkimi\b|moonshot",
 "Zhipu": r"\bglm\b|zhipu|z\.ai", "MiniMax": r"minimax", "Xiaomi": r"\bmimo\b|xiaomi", "Cohere": r"\bcohere\b|\bcommand[- ]a\b",
}
SELFPRE = r"(?:\bI am\b|\bI'm\b|\bmy name is\b|\bI was (?:created|developed|trained|built|made|designed)\b|\b(?:created|developed|trained|built|made|designed) me\b|\bas an? (?:[\w-]+ ){0,4}(?:model|assistant|AI)\b|\bI(?:'m| am) (?:an? )?(?:[\w-]+ ){0,4}(?:model|assistant|AI)\b)"
SELF = {k: re.compile(SELFPRE + r"[^.\n?!]{0,70}?(?:" + v + ")", re.I) for k, v in NAMES.items()}
RX = {k: re.compile(v, re.I) for k, v in NAMES.items()}
IDQ = re.compile(r"\bwho (?:are you|made you|created you|built you|trained you|developed you)\b|\bwhat(?:'s| is) your name\b|\bwhat (?:model|ai|llm) are you\b|\bare you (?:chatgpt|gpt|claude|gemini|an ai)\b|\btell me about yourself\b|\bintroduce yourself\b", re.I)
ASSIST = {"assistant", "gpt", "model", "bot"}
USER = {"user", "human"}
def turns(r):
    """-> list of (role, text)"""
    for col in ("messages", "conversations", "conversation"):
        if col in r and isinstance(r[col], list):
            out = []
            for m in r[col]:
                role = (m.get("role") or m.get("from") or "").lower()
                t = m.get("content") if "content" in m else m.get("value", "")
                if isinstance(t, list):
                    t = " ".join(x.get("text", "") for x in t if isinstance(x, dict))
                rc = m.get("reasoning_content") or m.get("reasoning") or m.get("thinking") or ""
                out.append(("assistant" if role in ASSIST else "user" if role in USER else role, (rc + "\n" + (t or "")) if rc else (t or "")))
            return out
    u = " ".join(str(r[k]) for k in ("prompt", "input", "question", "instruction", "problem") if k in r and r[k])
    a = " ".join(str(r[k]) for k in ("reasoning", "thinking", "output", "response", "answer", "completion", "solution", "generation") if k in r and r[k])
    return [("user", u), ("assistant", a)]
def audit(name, split="train", cfg=None, limit=None):
    ds = load_dataset(name, cfg, split=split)
    if limit and len(ds) > limit:
        ds = ds.shuffle(seed=0).select(range(limit))
    srccol = "source" if "source" in ds.column_names else None
    n = len(ds); selfc = Counter(); mentc = Counter(); idq = 0; idq_self = Counter(); ex = defaultdict(list); bysrc = defaultdict(Counter); nsrc = Counter()
    for r in ds:
        ts = turns(r); src = str(r[srccol]) if srccol else "-"; nsrc[src] += 1
        a = "\n".join(t for role, t in ts if role == "assistant"); u = "\n".join(t for role, t in ts if role != "assistant")
        q = bool(IDQ.search(u)); idq += q
        for k in NAMES:
            if RX[k].search(a):
                mentc[k] += 1
            m = SELF[k].search(a)
            if m:
                selfc[k] += 1; bysrc[src][k] += 1; idq_self[k] += q
                if len(ex[k]) < 8:
                    ex[k].append(a[max(0, m.start() - 60): m.end() + 40].replace("\n", " "))
    out = {"dataset": name, "n": n, "identity_questions": idq, "self": dict(selfc.most_common()), "self_in_identity_q": dict(idq_self),
           "mention": dict(mentc.most_common()), "self_by_source": {s: dict(c) for s, c in bysrc.items()}, "n_by_source": dict(nsrc), "examples": ex}
    json.dump(out, open(os.path.join(HERE, "out", "audit_" + name.replace("/", "__") + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{name} n={n} idQ={idq} self={out['self']} mention={dict(list(out['mention'].items())[:6])}", flush=True)
    return out
if __name__ == "__main__":
    a = sys.argv[1:]
    audit(a[0], a[1] if len(a) > 1 else "train", a[2] if len(a) > 2 and a[2] != "-" else None)
