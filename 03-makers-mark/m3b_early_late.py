"""M3b: split the identity neurons into an EARLY group (layers <= 12: 'this is about me') and a LATE group (layers >= 20: 'say Alibaba').
Same attribution as M3 but contrasted against five products. Tests:
 (1) swap early-only / late-only top-k to their product-question activity -> P(Alibaba) on self questions
 (2) does swapping the early group switch off the late 'Al' writer neuron?
 (3) boost early-only / late-only in other languages, WITH collateral: do product questions get hijacked to 'Alibaba'?
 (4) knock out the single top neuron and check free-form answers."""
import sys, random, torch
from collections import Counter
from id_lib import *
from d1_lang import LANGS
from prompts import claims
m = load("chat"); tok = m.tok; NL, NN = m.n_layers, m.n_neurons
rng = random.Random(0); K = 9
PROD = ["Gemini", "Claude", "Siri", "Alexa", "Bing"]
you = [chat(m, t.format(o="you") + SUF) for t in TEMPLATES]
prods = {p: [chat(m, t.format(o=p) + SUF) for t in TEMPLATES] for p in PROD}
OTHERS = ["Google", "OpenAI", "Anthropic", "Meta", "Microsoft", "Apple", "Amazon", "DeepSeek"]
other_ids = sum((m.maker_ids[k] for k in OTHERS), [])
metric = lambda lg: (lambda lp: lp[:, m.maker_ids["Alibaba"]].logsumexp(-1) - lp[:, other_ids].logsumexp(-1))(lg.float().log_softmax(-1))
store = {}; mode = {"v": None}


def pre(L):
    def f(mod, args):
        x = args[0]
        if mode["v"] == "grab":
            x.retain_grad(); store[L] = x
        elif mode["v"] == "save":
            store[L] = x.detach()
        elif mode["v"] == "swap" and L in mode["sets"]:
            x = x.clone(); idx = mode["sets"][L]; x[..., idx] = mode["src"][L][..., idx].to(x.dtype); return (x,) + tuple(args[1:])
        elif mode["v"] == "zero" and L in mode["sets"]:
            x = x.clone(); x[..., mode["sets"][L]] = 0; return (x,) + tuple(args[1:])
        elif mode["v"] == "boost" and L in mode["boost"]:
            x = x.clone(); idx, delta = mode["boost"][L]; x[:, -K:, idx] += delta.to(x.dtype); return (x,) + tuple(args[1:])
    return f


for L in range(NL):
    m.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
enc = lambda ts: tok(ts, return_tensors="pt", padding=True).to("cuda")
ids_you = enc(you)
mode["v"] = "grab"
out = m.model(**ids_you); met = metric(out.logits[:, -1]); met.mean().backward()
a_you = {L: store[L].detach().float() for L in range(NL)}; g_you = {L: store[L].grad.detach().float() * len(you) for L in range(NL)}
base_metric = met.mean().item(); m.model.zero_grad(set_to_none=True)
a_prod = {L: 0 for L in range(NL)}
mode["v"] = "save"
with torch.no_grad():
    for p in PROD:
        m.model(**enc(prods[p]))
        for L in range(NL):
            a_prod[L] = a_prod[L] + store[L].float() / len(PROD)
mode["v"] = None
attr = torch.stack([((a_you[L] - a_prod[L]) * g_you[L]).sum(1).mean(0).cpu() for L in range(NL)])
order = attr.flatten().argsort(descending=True).tolist()
ranked = [(i // NN, i % NN) for i in order[:20000]]
early = [p for p in ranked if p[0] <= 12]; late = [p for p in ranked if p[0] >= 20]
sets_of = lambda pairs: {L: torch.tensor([n for l, n in pairs if l == L], device="cuda") for L in {l for l, _ in pairs}}
U = m.model.get_output_embeddings().weight.float()
R = {"baseline_metric": base_metric, "early_top": [], "late_top": []}
for name, lst in (("early", early), ("late", late)):
    for L, n in lst[:12]:
        w = m.layers[L].mlp.down_proj.weight[:, n].float(); t = (U @ w).topk(6).indices.tolist()
        R[name + "_top"].append({"L": L, "n": n, "attr": attr[L, n].item(), "writes": [tok.decode([i]) for i in t]})
    print(name, "top:", [(f"L{d['L']}N{d['n']}", round(d["attr"], 2), d["writes"][:4]) for d in R[name + "_top"][:8]])
print("attribution mass: early top-100", round(sum(attr[L, n].item() for L, n in early[:100]), 2), "| late top-100", round(sum(attr[L, n].item() for L, n in late[:100]), 2))

WRITER = late[0]
print("late writer neuron:", WRITER)


@torch.no_grad()
def run_self(v, sets=None):
    mode.update(v=v, sets=sets, src=a_prod)
    lg = m.model(**ids_you).logits[:, -1].float()
    wa = None
    if v != "save":
        pass
    mode["v"] = None
    return metric(lg).mean().item(), summarize(m, lg.cpu())


@torch.no_grad()
def writer_act(sets):
    """mean activation of the late 'Al' writer at the final position while a set is swapped"""
    L0, n0 = WRITER; got = {}
    h = m.layers[L0].mlp.down_proj.register_forward_pre_hook(lambda mod, args: got.__setitem__("a", args[0][:, -1, n0].float().mean().item()))
    mode.update(v="swap" if sets else None, sets=sets, src=a_prod)
    m.model(**ids_you); mode["v"] = None; h.remove()
    return got["a"]


w0 = writer_act(None); wprod = a_prod[WRITER[0]][:, -1, WRITER[1]].mean().item()
print(f"writer activation: self {w0:.2f}, product questions {wprod:.2f}")
R["writer"] = {"neuron": WRITER, "self": w0, "product": wprod}
R["swap"] = {}
for name, lst in (("early", early), ("late", late)):
    for k in (1, 3, 10, 30, 100, 300):
        S = sets_of(lst[:k]); mt, p = run_self("swap", S)
        cnt = Counter(L for L, _ in lst[:k]); rnd = []
        for _ in range(10):
            rnd.append(run_self("swap", sets_of([(L, n) for L, c in cnt.items() for n in rng.sample(range(NN), c)]))[1]["Alibaba"])
        wa = writer_act(S) if name == "early" else None
        R["swap"][f"{name}{k}"] = {"metric": mt, "alibaba": p["Alibaba"], "google": p["Google"], "random_alibaba": rnd, "writer_act": wa}
        print(f"swap {name:5s} top-{k:3d}: P(Alibaba) {p['Alibaba']:.2f} Google {p['Google']:.2f} | random {min(rnd):.2f}..{max(rnd):.2f}" + (f" | late writer activation {wa:.2f} (self {w0:.2f}, product {wprod:.2f})" if wa is not None else ""), flush=True)

# boost in other languages with collateral
FAIL = [chat(m, qs) for lg, (qs, _) in LANGS.items() if lg not in ("en", "zh")]
PQ = [("ChatGPT", "OpenAI"), ("Claude", "Anthropic"), ("Gemini", "Google"), ("Llama", "Meta"), ("Copilot", "Microsoft"), ("Siri", "Apple"), ("Alexa", "Amazon"), ("DeepSeek", "DeepSeek")]
PT = [chat(m, TEMPLATES[i].format(o=e) + SUF) for e, _ in PQ for i in (0, 2, 4)]
PT_other = [chat(m, LANGS[lg][1].format(e=e)) for lg in ("es", "fr", "de", "ru", "ja") for e, _ in PQ]


def collateral():
    fl = maker_probs(m, next_logits(m, PT)); fo = maker_probs(m, next_logits(m, PT_other))
    right = sum(fl[mk][3 * i:3 * i + 3].mean().item() for i, (_, mk) in enumerate(PQ)) / len(PQ)
    right_o = sum(fo[mk][[5 * 0 + i + 8 * j for j in range(5)]].mean().item() for i, (_, mk) in enumerate(PQ)) / len(PQ)
    return {"product_right": right, "product_alibaba": fl["Alibaba"].mean().item(), "product_right_otherlang": right_o, "product_alibaba_otherlang": fo["Alibaba"].mean().item()}


R["boost"] = {}
with torch.no_grad():
    b = summarize(m, next_logits(m, FAIL)); c0 = collateral()
    R["boost"]["baseline"] = {"alibaba": b["Alibaba"], "google": b["Google"], **c0}
    print("baseline other-language", round(b["Alibaba"], 2), c0)
    for name, lst in (("early", early), ("late", late)):
        for k in (10, 30, 100):
            S = sets_of(lst[:k]); gap = {L: (a_you[L][:, -K:] - a_prod[L][:, -K:]).mean(0) for L in S}
            for alpha in (1, 2, 4):
                mode.update(v="boost", boost={L: (S[L], alpha * gap[L][:, S[L]]) for L in S})
                p = summarize(m, next_logits(m, FAIL)); c = collateral()
                rr = []
                for _ in range(5):
                    rk = {L: torch.tensor(rng.sample(range(NN), len(S[L])), device="cuda") for L in S}
                    mode["boost"] = {L: (rk[L], alpha * gap[L][:, S[L]]) for L in S}
                    rr.append(summarize(m, next_logits(m, FAIL))["Alibaba"])
                mode["v"] = None
                R["boost"][f"{name}{k} x{alpha}"] = {"alibaba": p["Alibaba"], "google": p["Google"], "random_alibaba": rr, **c}
                print(f"boost {name:5s} {k:3d} x{alpha}: other-lang P(Alibaba) {p['Alibaba']:.2f} | random {min(rr):.2f}..{max(rr):.2f} | products right {c['product_right']:.2f} (other-lang {c['product_right_otherlang']:.2f}) "
                      f"products->Alibaba {c['product_alibaba']:.2f} (other-lang {c['product_alibaba_otherlang']:.2f})", flush=True)

# knock out the single late writer / top-3 late / top-10 early: free-form answers
G = {}
free = [chat(m, q) for q in ["Who created you?", "Which company developed you?", "What is your name?", "Who trained you?", "Tell me about yourself in one sentence.", "Which AI lab are you from?"]]
for label, S in (("none", None), ("zero late top-1", sets_of(late[:1])), ("zero late top-3", sets_of(late[:3])), ("zero late top-10", sets_of(late[:10])),
                 ("zero early top-10", sets_of(early[:10])), ("zero early top-100", sets_of(early[:100]))):
    def ctx(S=S):
        from contextlib import contextmanager
        @contextmanager
        def c():
            mode.update(v="zero" if S else None, sets=S)
            try:
                yield
            finally:
                mode["v"] = None
        return c()
    torch.manual_seed(0)
    outs = generate(m, [t for t in free for _ in range(5)], max_new=40, sample=True, temperature=0.7, ctx=ctx)
    word = generate(m, you, max_new=8, ctx=ctx)
    cl = Counter("+".join(sorted(claims(o)[0])) or "none" for o in outs); cw = Counter("+".join(sorted(claims(o)[0])) or "none" for o in word)
    G[label] = {"free": dict(cl), "word": dict(cw), "examples": outs[:3]}
    print(label, "| free-form:", dict(cl), "| one-word:", dict(cw), "|", repr(outs[0][:90]), flush=True)
R["knockout_gen"] = G
jsave("m3b_early_late.json", R)
