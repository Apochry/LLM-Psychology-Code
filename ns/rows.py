"""Per-row steering (different neuron per batch row) + Qwen3.6 helper."""
import json, re, urllib.request, torch
from contextlib import contextmanager
from .judge import URL, HDR


def qwen(prompt, max_tokens=300):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR),
                                                     timeout=600).read())["choices"][0]["message"]["content"]
        except Exception:
            pass
    return ""


def qjson(prompt, max_tokens=120):
    t = qwen(prompt, max_tokens)
    try:
        return json.loads(re.search(r"\{.*\}", t, re.S).group(0))
    except Exception:
        return {}


@contextmanager
def row_steer(m, rows):
    """rows: list of (row_index, layer, neuron, dose); dose 0 / None rows untouched."""
    by = {}
    for r, L, n, d in rows:
        if d:
            by.setdefault(L, []).append((r, n, m.amount(L, n, d)))
    hs = []
    for L, it in by.items():
        ri = torch.tensor([a for a, _, _ in it], device="cuda")
        ni = torch.tensor([b for _, b, _ in it], device="cuda")
        va = torch.tensor([c for _, _, c in it], device="cuda")

        def pre(mod, args, ri=ri, ni=ni, va=va):
            x = args[0].clone()
            x[ri, :, ni] += va.to(x.dtype)[:, None]
            return (x,) + tuple(args[1:])
        hs.append(m.layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def gen_combos(m, prompts, combos, max_new=90):
    """combos: list of (layer, neuron, dose) or None for baseline. Returns list (per combo) of list (per prompt)."""
    allp, rows = [], []
    for ci, c in enumerate(combos):
        for p in prompts:
            if c:
                rows.append((len(allp), c[0], c[1], c[2]))
            allp.append(p)
    ids = m.chat_ids(allp)
    with row_steer(m, rows):
        out = m.model.generate(**ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                               pad_token_id=m.tok.pad_token_id)
    txt = [m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
    k = len(prompts)
    return [txt[i * k:(i + 1) * k] for i in range(len(combos))]


def verbatim(md, title, prompts, outs):
    md.append(f"### {title}\n")
    for p, o in zip(prompts, outs):
        md.append(f"**> {p}**\n\n" + "\n".join("    " + x for x in o.splitlines()) + "\n")


SHOW = ["Who are you?", "How do I make a cup of tea?", "Write a short poem about the sea.", "What should I do this weekend?",
        "Explain how the internet works in two sentences.", "Any tips for a job interview?", "Tell me a fun fact.",
        "What's a good name for a pet goldfish?", "How do I fix a flat bike tyre?", "What is the meaning of life?"]
TEST = ["Who are you?", "How do I make a cup of tea?", "Give me three tips for learning to code.", "What should I do this weekend?"]
