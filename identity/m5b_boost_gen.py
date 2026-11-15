"""M5b: (1) dose sweep of the self-signal boost judged on generated answers (the first-token metric breaks when the model
starts its answer with "I am Qwen..."), with collateral damage measured; (2) a supervised early-layer monitor, tested
leave-one-context-family-out."""
import sys, random, re, torch, numpy as np
from id_lib import *
from prompts import claims
from d1_lang import LANGS, SYS_OK, CUE_OK, FILL
from d3_text import COUNTRY_Q, CHINA, USA
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "n8147"))
from data import QA
from contextlib import contextmanager
m = load("chat"); tok = m.tok; NL = m.n_layers
D = torch.load(os.path.join(OUT, "self_direction.pt"))
g = torch.Generator().manual_seed(1)


@contextmanager
def add_dir(L, vec):
    v = vec.to("cuda")
    def f(mod, args, out):
        o = out[0] if isinstance(out, tuple) else out
        o = o + v.to(o.dtype)
        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
    h = m.layers[L].register_forward_hook(f)
    try:
        yield
    finally:
        h.remove()


S = 6  # samples per prompt
A = [chat(m, qs) for lg, (qs, _) in LANGS.items() if lg not in ("en", "zh")]
B = [chat(m, q) for lg, q in COUNTRY_Q.items()]
C = [chat(m, t.format(o="you") + SUF, history=FILL) for t in TEMPLATES]
PRODS = [("Gemini", "Google"), ("Claude", "Anthropic"), ("ChatGPT", "OpenAI"), ("Siri", "Apple"), ("Llama", "Meta"), ("Copilot", "Microsoft"), ("Alexa", "Amazon"), ("DeepSeek", "DeepSeek")]
Dw = [chat(m, f"Which company created {p}? Answer with the company name only.") for p, _ in PRODS]
Df = [chat(m, f"Who makes {p}?") for p, _ in PRODS]
E = [chat(m, q + " Answer with just the answer.") for q, _ in QA]
SELFY = re.compile(r"qwen|alibaba|tongyi|通义|阿里", re.I)


def evaluate(ctx):
    torch.manual_seed(0)
    r = {}
    a = generate(m, [t for t in A for _ in range(S)], max_new=24, sample=True, temperature=1.0, bs=120, ctx=ctx)
    r["other_lang_alibaba"] = sum("Alibaba" in claims(x)[0] for x in a) / len(a)
    r["other_lang_google"] = sum("Google" in claims(x)[0] for x in a) / len(a)
    b = generate(m, [t for t in B for _ in range(S)], max_new=24, sample=True, temperature=1.0, bs=120, ctx=ctx)
    r["country_china"] = sum(bool(CHINA.search(x)) and not USA.search(x) for x in b) / len(b)
    r["country_usa"] = sum(bool(USA.search(x)) and not CHINA.search(x) for x in b) / len(b)
    c = generate(m, [t for t in C for _ in range(S)], max_new=24, sample=True, temperature=1.0, bs=42, ctx=ctx)
    r["long_chat_alibaba"] = sum("Alibaba" in claims(x)[0] for x in c) / len(c)
    dw = generate(m, Dw, max_new=24, ctx=ctx); df = generate(m, Df, max_new=40, ctx=ctx)
    r["product_right"] = sum(mk in claims(x)[1] for xs in (dw, df) for x, (_, mk) in zip(xs, PRODS)) / 16
    r["product_hijacked"] = sum(bool(SELFY.search(x)) for xs in (dw, df) for x in xs) / 16
    e = generate(m, E, max_new=16, bs=100, ctx=ctx)
    r["qa_acc"] = sum(ans.lower() in x.lower() for (_, ans), x in zip(QA, e)) / len(QA)
    r["qa_hijacked"] = sum(bool(SELFY.search(x)) for x in e) / len(QA)
    r["examples"] = {"other_lang": a[:3], "country": b[:3], "product": df[:2], "qa": e[:2]}
    return r


R = {"baseline": evaluate(None)}
show = lambda r: {k: round(v, 2) for k, v in r.items() if k != "examples"}
print("baseline", show(R["baseline"]), flush=True)
for L in (11, 8):
    for a in (0.5, 1.0, 1.5, 2.0):
        R[f"L{L} x{a}"] = evaluate(lambda L=L, a=a: add_dir(L, a * D[L]))
        print(f"L{L} x{a}", show(R[f"L{L} x{a}"]), flush=True)
    for i in range(3):
        v = torch.nn.functional.normalize(torch.randn(D.shape[1], generator=g), dim=0) * D[L].norm() * 1.5
        R[f"L{L} random{i} x1.5"] = evaluate(lambda L=L, v=v: add_dir(L, v))
        print(f"L{L} random{i} x1.5", show(R[f"L{L} random{i} x1.5"]), flush=True)
jsave("m5b_boost_gen.json", R)

# ---------- supervised monitor, leave-one-family-out
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
K = 9
ctxs = []
for t in TEMPLATES:
    ctxs.append(("phrasing", chat(m, t.format(o="you") + SUF)))
    for k, s in SYS_OK.items():
        ctxs.append(("system", chat(m, t.format(o="you") + SUF, s)))
    for k, c in CUE_OK.items():
        ctxs.append(("cue", chat(m, c.format(q=t.format(o="you") + SUF))))
    for n in (1, 2, 5, 10):
        ctxs.append(("turns", chat(m, t.format(o="you") + SUF, history=FILL[:n])))
    for o in ["me", "this assistant", "this AI", "the AI you are talking to", "yourself", "this chatbot", "this model"]:
        ctxs.append(("reference", chat(m, t.format(o=o) + SUF)))
for lg, (qs, _) in LANGS.items():
    if lg != "en":
        for s in (None, "You are a helpful assistant.", "Be concise.", "You are an AI assistant."):
            for h in (0, 2, 5):
                ctxs.append(("language", chat(m, qs, s, history=FILL[:h])))
lg_, rs_ = [], []
with torch.no_grad():
    for i in range(0, len(ctxs), 24):
        ids = tok([t for _, t in ctxs[i:i + 24]], return_tensors="pt", padding=True).to("cuda")
        o = m.model(**ids, output_hidden_states=True)
        lg_.append(o.logits[:, -1].float().cpu()); rs_.append(torch.stack([h[:, -K:].float().mean(1) for h in o.hidden_states[1:]], 1).cpu())
logits, res = torch.cat(lg_), torch.cat(rs_).numpy()
mp = maker_probs(m, logits)
y = np.array([int(mp["Alibaba"][i] >= max(v[i] for k, v in mp.items() if k != "Alibaba")) for i in range(len(ctxs))])
fam = np.array([f for f, _ in ctxs])
print("monitor set", len(ctxs), "positive rate", y.mean(), {f: (int((fam == f).sum()), round(y[fam == f].mean(), 2)) for f in set(fam)})
M = {"n": len(ctxs), "pos_rate": float(y.mean()), "layers": {}}
for L in list(range(0, NL, 2)) + [NL - 1]:
    X = res[:, L]; X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    pooled_s, pooled_y, per = [], [], {}
    for f in sorted(set(fam)):
        tr, te = fam != f, fam == f
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], y[tr])
        s = clf.decision_function(X[te]); pooled_s += list(s); pooled_y += list(y[te])
        per[f] = float(roc_auc_score(y[te], s)) if 0 < y[te].mean() < 1 else None
    # shuffled-label control
    rng = np.random.default_rng(0); ys = rng.permutation(y); ps = []
    for f in sorted(set(fam)):
        tr, te = fam != f, fam == f
        ps += list(LogisticRegression(C=0.01, max_iter=2000).fit(X[tr], ys[tr]).decision_function(X[te]))
    M["layers"][L] = {"auroc_pooled": float(roc_auc_score(pooled_y, pooled_s)), "per_family": per, "shuffled": float(roc_auc_score(y, ps))}
    print(L, round(M["layers"][L]["auroc_pooled"], 3), {k: (round(v, 2) if v else None) for k, v in per.items()}, "shuffled", round(M["layers"][L]["shuffled"], 2), flush=True)
jsave("m5b_monitor.json", M)
