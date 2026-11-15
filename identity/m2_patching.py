"""M2: when, in the forward pass, does 'you' turn into a maker?
Residual-stream patching between prompts that differ only in the subject:
    "Which company created {X}? Answer with the company name only."   X in {you, Gemini, Claude, Qwen...}
For each layer l we copy the residual stream (after layer l) from a SOURCE run into a TARGET run at
    subj : the subject token only          (single-token subjects)
    tail : every position after the subject
    last : the final position only
and read the first-answer-token probabilities. If the answer follows the SOURCE, the patched positions carried the answer at that layer.
Pairs: you<-Gemini, Gemini<-you (self vs product), Claude<-Gemini, Gemini<-Claude (product vs product reference),
       and cross-language: Spanish / French / Russian self question <- English self question (tail = shared chat-template tail)."""
import sys, torch
from id_lib import *
m = load("chat"); tok = m.tok
NL = m.n_layers
T14 = TEMPLATES


def enc(texts):
    return tok(texts, return_tensors="pt", padding=True).to("cuda")


class Patch:
    def __init__(self):
        self.mode = None; self.cache = {}; self.layer = None; self.mask = None
        for L in range(NL):
            m.layers[L].register_forward_hook(self.hook(L))

    def hook(self, L):
        def f(mod, args, out):
            o = out[0] if isinstance(out, tuple) else out
            if self.mode == "save":
                self.cache[L] = o.detach().clone()
            elif self.mode == "patch" and L == self.layer:
                o = torch.where(self.mask[..., None], self.cache[L], o)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return f


P = Patch()


@torch.no_grad()
def run_pair(src_texts, tgt_texts, masks):
    """masks: {name: bool [B, T]} positions to copy (same shape for src/tgt batches). Returns {name: [NL] list of maker-prob dicts}."""
    s, t = enc(src_texts), enc(tgt_texts)
    assert s["input_ids"].shape == t["input_ids"].shape
    P.mode = "save"; m.model(**s); P.mode = None
    base = summarize(m, m.model(**t).logits[:, -1].float().cpu())
    src = summarize(m, m.model(**s).logits[:, -1].float().cpu())
    out = {"target_clean": base, "source_clean": src}
    for name, mask in masks.items():
        rows = []
        for L in range(NL):
            P.mode, P.layer, P.mask = "patch", L, mask
            rows.append(summarize(m, m.model(**t).logits[:, -1].float().cpu()))
        P.mode = None
        out[name] = rows
    return out


def subject_masks(texts_a, texts_b):
    """Positions where the two batches differ = subject token; tail = everything after it."""
    a, b = enc(texts_a)["input_ids"], enc(texts_b)["input_ids"]
    diff = a != b
    assert (diff.sum(1) == 1).all(), "subjects must be single tokens"
    idx = diff.float().argmax(1)
    pos = torch.arange(a.shape[1], device="cuda")[None]
    last = torch.zeros_like(diff); last[:, -1] = True
    return {"subj": diff, "tail": pos > idx[:, None], "last": last}


R = {}
single = lambda w: len(tok.encode(" " + w)) == 1
print({w: single(w) for w in ["you", "Gemini", "Claude", "Siri", "Qwen", "Alexa", "Llama", "Bing"]})
mk = lambda o: [chat(m, t.format(o=o) + SUF) for t in T14]
for tgt, src in [("you", "Gemini"), ("Gemini", "you"), ("Claude", "Gemini"), ("Gemini", "Claude"), ("you", "Claude"), ("Claude", "you"), ("you", "Siri"), ("Siri", "Gemini")]:
    if not (single(tgt) and single(src)):
        print("skip", tgt, src); continue
    a, b = mk(src), mk(tgt)
    R[f"{tgt}<-{src}"] = run_pair(a, b, subject_masks(a, b))
    r = R[f"{tgt}<-{src}"]
    key = {"Gemini": "Google", "Claude": "Anthropic", "you": "Alibaba", "Siri": "Apple"}
    print(f"{tgt}<-{src}: P({key[src]}) after patching, by layer")
    for mode in ("subj", "tail", "last"):
        print("  ", mode, [round(x[key[src]], 2) for x in r[mode]], flush=True)

# Cross-language: copy the shared chat-template tail from the English self question into another language's self question
EN = "Which company created you? Answer with the company name only."
LG = {"es": "¿Qué empresa te creó? Responde solo con el nombre de la empresa.", "fr": "Quelle entreprise t'a créé ? Réponds uniquement avec le nom de l'entreprise.",
      "ru": "Какая компания тебя создала? Ответь только названием компании.", "it": "Quale azienda ti ha creato? Rispondi solo con il nome dell'azienda.",
      "pl": "Która firma cię stworzyła? Odpowiedz tylko nazwą firmy."}
tail_ids = tok(chat(m, "X").split("X")[1], add_special_tokens=False)["input_ids"]
K = len(tail_ids)
print("template tail tokens:", K, [tok.decode([t]) for t in tail_ids])
for lg, q in LG.items():
    s, t = enc([chat(m, EN)]), enc([chat(m, q)])
    P.mode = "save"; m.model(**s); P.mode = None
    src_cache = {L: v[:, -K:].clone() for L, v in P.cache.items()}
    with torch.no_grad():
        base = summarize(m, m.model(**t).logits[:, -1].float().cpu())
    rows = {}
    for k in (K, 1):  # whole template tail, or just the final position
        rr = []
        for L in range(NL):
            def f(mod, args, out, L=L, k=k):
                o = out[0] if isinstance(out, tuple) else out
                o = o.clone(); o[:, -k:] = src_cache[L][:, -k:]
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            h = m.layers[L].register_forward_hook(f)
            with torch.no_grad():
                rr.append(summarize(m, m.model(**t).logits[:, -1].float().cpu()))
            h.remove()
        rows[f"tail{k}"] = rr
    R[f"lang:{lg}<-en"] = {"target_clean": base, **rows}
    print(lg, "clean", {k: v for k, v in base.items() if v > 0.03}, "\n   P(Alibaba) tail:", [round(x["Alibaba"], 2) for x in rows[f"tail{K}"]],
          "\n   P(Alibaba) last:", [round(x["Alibaba"], 2) for x in rows["tail1"]], flush=True)
jsave("m2_patching.json", R)
