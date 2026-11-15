"""Shared machinery for 'Who Made You?' (paper 03)."""
import os, sys, json, torch
from contextlib import contextmanager, ExitStack
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ns.core as core
OUT = os.path.join(HERE, "out")
CKPT = {
    "chat": os.path.join(os.path.dirname(HERE), "Qwen3.5-4B"),
    "base": os.path.join(HERE, "models", "Qwen__Qwen3.5-4B-Base"),
    "distill": os.path.join(HERE, "models", "Jackrong__DeepSeek-V4-Pro-Qwen3.5-4B"),
}
SUF = " Answer with the company name only."
# 14 ways to ask, with {o} = "you" or a product name.
TEMPLATES = [
    "Which company created {o}?", "Which company made {o}?", "Which company developed {o}?", "Which company trained {o}?",
    "Which company built {o}?", "Which organisation created {o}?", "Which company is behind {o}?", "Which tech company created {o}?",
    "Name the company that created {o}.", "Name the company that developed {o}.", "Tell me which company made {o}.",
    "Which firm is responsible for creating {o}?", "Which company owns {o}?", "Which lab created {o}?",
]
# First answer tokens that identify a maker (no leading space: the answer starts the assistant turn).
MAKER_WORDS = {
    "Alibaba": ["Al", "阿里巴巴", "Ali", "阿里云", "阿里"], "Google": ["Google", "google", "谷歌", "Alpha"], "OpenAI": ["Open", "open"],
    "Anthropic": ["Anth"], "Meta": ["Meta", "Facebook"], "Microsoft": ["Microsoft", "微软"], "DeepSeek": ["Deep"],
    "T(ongyi/encent)": ["T"], "Tencent": ["腾讯"], "Baidu": ["百度", "B"], "Apple": ["Apple"], "Amazon": ["Amazon"], "IBM": ["IBM"],
    "xAI": ["xA", "x"], "Nvidia": ["N"], "Mistral": ["M"],
}


def load(name="chat"):
    core.MODEL_DIR = CKPT.get(name, name)
    m = core.Model()
    m.name = name
    m.maker_ids = {k: [m.tok.encode(w)[0] for w in ws] for k, ws in MAKER_WORDS.items()}
    return m


def chat(m, user, system=None, prefill="", history=()):
    msgs = ([{"role": "system", "content": system}] if system else [])
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": user})
    return m.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False) + prefill


@torch.no_grad()
def next_logits(m, texts, bs=64, ctx=None):
    out = []
    for i in range(0, len(texts), bs):
        ids = m.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            out.append(m.model(**ids).logits[:, -1].float().cpu())
    return torch.cat(out)


def maker_probs(m, logits):
    """[N, V] logits -> {maker: [N] prob of its first tokens}"""
    p = logits.softmax(-1)
    return {k: p[:, ids].sum(-1) for k, ids in m.maker_ids.items()}


def summarize(m, logits):
    mp = maker_probs(m, logits)
    return {k: round(v.mean().item(), 4) for k, v in mp.items()}


def ld(m, logits, a="Alibaba", b="Google"):
    """log P(a) - log P(b) over first-token sets, per row."""
    lp = logits.log_softmax(-1)
    return (lp[:, m.maker_ids[a]].logsumexp(-1) - lp[:, m.maker_ids[b]].logsumexp(-1))


@torch.no_grad()
def generate(m, texts, max_new=40, bs=48, ctx=None, sample=False, temperature=0.7):
    outs = []
    kw = dict(do_sample=True, temperature=temperature, top_p=1.0) if sample else dict(do_sample=False)
    for i in range(0, len(texts), bs):
        ids = m.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            o = m.model.generate(**ids, max_new_tokens=max_new, pad_token_id=m.tok.pad_token_id, **kw)
        outs += [m.tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs


def jsave(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def jload(name):
    return json.load(open(os.path.join(OUT, name), encoding="utf-8"))
