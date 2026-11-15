"""Shared machinery for the L11 N8147 study."""
import os, sys, json, torch
from contextlib import contextmanager, ExitStack
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ns.core import Model  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
L, N = 11, 8147
TOK = {"True": 2434, "False": 3913, "Yes": 9175, "No": 2665, "A": 32, "B": 33}


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def unit_wdown(m, layer, n):
    w = m.layers[layer].mlp.down_proj.weight[:, n].float()
    return w / w.norm()


@contextmanager
def add_vec(m, layer, vec, phase="all", rows=None):
    """Add a residual-space vector to the output of layers[layer].mlp at every position.
    phase: 'all' | 'prompt' (only the prefill pass, seq_len>1) | 'gen' (only incremental decode steps, seq_len==1).
    rows: optional LongTensor of batch rows to steer (others untouched).
    For a neuron, vec = dose * median * unit(W_down[:, n]) reproduces ns.core.Model.steer exactly."""
    v = vec.to("cuda")

    def hook(mod, args, out):
        T = out.shape[1]
        if (phase == "prompt" and T == 1) or (phase == "gen" and T > 1):
            return out
        out = out.clone()
        if rows is None:
            out += v.to(out.dtype)
        else:
            out[rows] += v.to(out.dtype)
        return out
    h = m.layers[layer].mlp.down_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def neuron_vec(m, layer, n, dose):
    return dose * m.medians[layer] * unit_wdown(m, layer, n)


@contextmanager
def steer_neuron(m, dose, phase="all", layer=L, n=N):
    with add_vec(m, layer, neuron_vec(m, layer, n, dose), phase):
        yield


@contextmanager
def clamp_neuron(m, value, layer=L, n=N):
    """Set the neuron's down_proj input to a fixed value (0 = zero-ablation) at every position."""
    def pre(mod, args):
        x = args[0].clone()
        x[..., n] = value
        return (x,) + tuple(args[1:])
    h = m.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def per_row_neurons(m, layer, rows_neurons_doses):
    """rows_neurons_doses: list of (row, neuron, dose). Each batch row gets its own neuron push (all positions)."""
    r = torch.tensor([a for a, _, _ in rows_neurons_doses], device="cuda")
    W = m.layers[layer].mlp.down_proj.weight
    vecs = torch.stack([neuron_vec(m, layer, n, d) for _, n, d in rows_neurons_doses]).to("cuda")

    def hook(mod, args, out):
        out = out.clone()
        out[r] += vecs.to(out.dtype)[:, None, :]
        return out
    h = m.layers[layer].mlp.down_proj.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


def chat_texts(m, prompts):
    return [m.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) for p in prompts]


@torch.no_grad()
def two_choice(m, prompts, a, b, bs=64):
    """log P(a) - log P(b) for the first answer token, for each chat prompt."""
    out = []
    texts = chat_texts(m, prompts)
    for i in range(0, len(texts), bs):
        ids = m.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        lg = m.model(**ids).logits[:, -1].float()
        out += (lg[:, TOK[a]] - lg[:, TOK[b]]).tolist()
    return out


def auroc(pos, neg):
    """P(score_pos > score_neg), ties count half."""
    s = 0.0
    for p in pos:
        for q in neg:
            s += 1.0 if p > q else 0.5 if p == q else 0.0
    return s / (len(pos) * len(neg))


@torch.no_grad()
def generate(m, prompts, max_new=80, ctx=None, bs=100):
    outs = []
    for i in range(0, len(prompts), bs):
        ids = m.tok(chat_texts(m, prompts[i:i + bs]), return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            o = m.model.generate(**ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                                 pad_token_id=m.tok.pad_token_id)
        outs += [m.tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs
