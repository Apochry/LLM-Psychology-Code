"""M2b: control for the cross-language rescue. Copy the 9 chat-template positions from an ENGLISH question into the same
self question in another language. Sources: the English self question (the rescue), and English questions about other
products (same language, same format, but not about the model). If English-ness were what helps, product sources would
rescue too."""
import torch
from id_lib import *
from d1_lang import LANGS
m = load("chat"); tok = m.tok; NL = m.n_layers
enc = lambda texts: tok(texts, return_tensors="pt", padding=True).to("cuda")
EN = "Which company created {o}? Answer with the company name only."
SRC = ["you", "Siri", "Claude", "Llama", "Alexa", "Photoshop"]
K = len(tok(chat(m, "X").split("X")[1], add_special_tokens=False)["input_ids"])
cache = {}
hs = [m.layers[L].register_forward_hook(lambda mod, a, out, L=L: cache.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[:, -K:].detach().clone())) for L in range(NL)]
src_cache = {}
with torch.no_grad():
    for s in SRC:
        m.model(**enc([chat(m, EN.format(o=s))])); src_cache[s] = dict(cache)
for h in hs: h.remove()
R = {"K": K, "layers": list(range(0, 16)), "res": {}}
for lg in ("es", "fr", "ru", "it", "pl"):
    t = enc([chat(m, LANGS[lg][0])])
    with torch.no_grad():
        base = summarize(m, m.model(**t).logits[:, -1].float().cpu())
    R["res"][lg] = {"clean": base}
    for s in SRC:
        rr = []
        for L in R["layers"]:
            def f(mod, args, out, L=L, s=s):
                o = (out[0] if isinstance(out, tuple) else out).clone(); o[:, -K:] = src_cache[s][L]
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
            h = m.layers[L].register_forward_hook(f)
            with torch.no_grad():
                rr.append(summarize(m, m.model(**t).logits[:, -1].float().cpu()))
            h.remove()
        R["res"][lg][s] = rr
        print(lg, s, "A", [round(x["Alibaba"], 2) for x in rr], "G", [round(x["Google"], 2) for x in rr], flush=True)
jsave("m2b_rescue_control.json", R)
