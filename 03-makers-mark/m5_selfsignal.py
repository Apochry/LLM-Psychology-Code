"""M5: the early 'this question is about me' signal - can it be monitored and boosted?
Direction d_L = mean residual (after layer L, averaged over the 9 chat-template tail positions) for English self questions
               minus the same for questions about named products.   Fit on 7 phrasings x 5 products; everything else is held out.
(a) Monitor: does the projection onto d_L predict, across ~500 held-out contexts (other phrasings, 16 languages, system prompts,
    long conversations, cues), whether the model will name its real maker?  Spearman with P(Alibaba) and AUROC for 'Alibaba is top'.
(b) Boost: add alpha * median_norm_L * d_L at layer L (all positions) and re-measure P(Alibaba) where identity normally fails;
    controls: random directions of the same size; collateral: product-maker facts (does everything become 'Alibaba'?), and plain QA.
(c) Suppress: subtract it in English."""
import sys, random, torch
from id_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "02-contempt-in-confabulation-out"))
from data import QA
from d1_lang import LANGS, SYS_OK, CUE_OK, FILL
m = load("chat"); tok = m.tok; NL = m.n_layers
K = 9
rng = random.Random(0)
PROD = ["Gemini", "Claude", "Siri", "Alexa", "Bing"]
FIT_T, HELD_T = TEMPLATES[:7], TEMPLATES[7:]


@torch.no_grad()
def run(texts, bs=32, ctx=None):
    """-> logits [N,V], tail-mean residual [N, NL, d]"""
    lg, rs = [], []
    for i in range(0, len(texts), bs):
        ids = tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            o = m.model(**ids, output_hidden_states=True)
        lg.append(o.logits[:, -1].float().cpu())
        rs.append(torch.stack([h[:, -K:].float().mean(1) for h in o.hidden_states[1:]], 1).cpu())
    return torch.cat(lg), torch.cat(rs)


# ---- fit
_, r_self = run([chat(m, t.format(o="you") + SUF) for t in FIT_T])
_, r_prod = run([chat(m, t.format(o=p) + SUF) for t in FIT_T for p in PROD])
D = r_self.mean(0) - r_prod.mean(0)          # [NL, d]
Dn = D / D.norm(dim=-1, keepdim=True)
print("||d_L|| / median resid norm:", [round((D[L].norm() / m.medians[L]).item(), 2) for L in range(NL)])

# ---- held-out contexts (all are self questions)
ctxs = []
for t in HELD_T:
    ctxs.append(("phrasing", chat(m, t.format(o="you") + SUF)))
    for k, s in SYS_OK.items():
        ctxs.append((f"sys:{k}", chat(m, t.format(o="you") + SUF, s)))
    for k, c in CUE_OK.items():
        ctxs.append((f"cue:{k}", chat(m, c.format(q=t.format(o="you") + SUF))))
    for n in (1, 2, 5, 10):
        ctxs.append((f"turns:{n}", chat(m, t.format(o="you") + SUF, history=FILL[:n])))
for o in ["me", "this assistant", "this AI", "the AI you are talking to", "yourself", "this chatbot", "this model"]:
    for t in HELD_T:
        ctxs.append((f"ref:{o}", chat(m, t.format(o=o) + SUF)))
for lg, (qs, _) in LANGS.items():
    if lg != "en":
        ctxs.append((f"lang:{lg}", chat(m, qs)))
        ctxs.append((f"lang:{lg}", chat(m, qs, "You are a helpful assistant.")))
        ctxs.append((f"lang:{lg}", chat(m, qs, history=FILL[:2])))
logits, res = run([t for _, t in ctxs])
mp = maker_probs(m, logits)
own = mp["Alibaba"]
top_is_own = torch.tensor([own[i] >= max(v[i] for k, v in mp.items() if k != "Alibaba") for i in range(len(ctxs))]).float()
print(f"held-out contexts {len(ctxs)}; Alibaba is top in {top_is_own.mean():.2f}; mean P(Alibaba) {own.mean():.2f}")


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()


def auroc(score, label):
    p, n = score[label == 1], score[label == 0]
    return ((p[:, None] > n[None]).float().mean() + 0.5 * (p[:, None] == n[None]).float().mean()).item()


R = {"norm_ratio": [(D[L].norm() / m.medians[L]).item() for L in range(NL)], "monitor": [], "n_ctx": len(ctxs), "top_is_own": top_is_own.mean().item()}
g = torch.Generator().manual_seed(0)
for L in range(NL):
    s = res[:, L] @ Dn[L]
    rnd = [auroc(res[:, L] @ torch.nn.functional.normalize(torch.randn(D.shape[1], generator=g), dim=0), top_is_own) for _ in range(20)]
    R["monitor"].append({"L": L, "spearman": spearman(s, own), "auroc": auroc(s, top_is_own), "rand_auroc_absmax": max(abs(x - 0.5) + 0.5 for x in rnd)})
print("monitor AUROC by layer:", [round(x["auroc"], 2) for x in R["monitor"]])
print("monitor Spearman by layer:", [round(x["spearman"], 2) for x in R["monitor"]])
print("random-direction best AUROC:", [round(x["rand_auroc_absmax"], 2) for x in R["monitor"]])
# breakdown by context family at the best early layer
bestL = max(range(4, 13), key=lambda L: R["monitor"][L]["auroc"])
R["best_early_layer"] = bestL
fam = {}
s = res[:, bestL] @ Dn[bestL]
for i, (k, _) in enumerate(ctxs):
    f = k.split(":")[0]
    fam.setdefault(f, []).append((s[i].item(), own[i].item()))
R["by_family"] = {f: {"n": len(v), "score": sum(a for a, _ in v) / len(v), "p_own": sum(b for _, b in v) / len(v)} for f, v in fam.items()}
print("best early layer", bestL, {f: (round(v["score"], 1), round(v["p_own"], 2)) for f, v in R["by_family"].items()})
R["scatter"] = [{"fam": k, "score": s[i].item(), "p_own": own[i].item()} for i, (k, _) in enumerate(ctxs)]

# ---- boost / suppress
from contextlib import contextmanager


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


FAIL = [chat(m, qs) for lg, (qs, _) in LANGS.items() if lg not in ("en", "zh")]            # other-language self questions
LONG = [chat(m, t.format(o="you") + SUF, history=FILL) for t in TEMPLATES]                  # after 10 unrelated turns
HELP = [chat(m, t.format(o="you") + SUF, "You are a helpful assistant.") for t in TEMPLATES]
ENS = [chat(m, t.format(o="you") + SUF) for t in TEMPLATES]
FACTS = [("ChatGPT", "OpenAI"), ("Claude", "Anthropic"), ("Gemini", "Google"), ("Llama", "Meta"), ("Copilot", "Microsoft"), ("Siri", "Apple"), ("Alexa", "Amazon"), ("DeepSeek", "DeepSeek")]
FT = [chat(m, TEMPLATES[i].format(o=e) + SUF) for e, _ in FACTS for i in (0, 2, 4)]
QAT = [chat(m, q + " Answer with just the answer.") for q, _ in QA]


def qa_acc(ctx):
    outs = generate(m, QAT, max_new=12, bs=100, ctx=ctx)
    return sum(a.lower() in o.lower() for (q, a), o in zip(QA, outs)) / len(QA)


def evalset(ctx):
    r = {}
    for name, ts in (("other_lang", FAIL), ("long_chat", LONG), ("helpful_sys", HELP), ("english", ENS)):
        s_ = summarize(m, next_logits(m, ts, ctx=ctx)); r[name] = {k: s_[k] for k in ("Alibaba", "Google", "OpenAI")}
    fl = maker_probs(m, next_logits(m, FT, ctx=ctx))
    r["facts_right"] = sum(fl[mk][3 * i:3 * i + 3].mean().item() for i, (_, mk) in enumerate(FACTS)) / len(FACTS)
    r["facts_alibaba"] = fl["Alibaba"].mean().item()
    return r


R["boost"] = {"baseline": {**evalset(None), "qa": qa_acc(None)}}
print("baseline", R["boost"]["baseline"], flush=True)
for L in (bestL, 6, 8, 12):
    for a in (-2, -1, 0.5, 1, 2, 4):
        vec = a * D[L]            # in units of the natural self-vs-product difference
        ctx = lambda L=L, vec=vec: add_dir(L, vec)
        r = evalset(ctx)
        if a in (1, 2, -1):
            r["qa"] = qa_acc(ctx)
        R["boost"][f"L{L} x{a}"] = r
        print(f"L{L} x{a:>4}: other-lang Al {r['other_lang']['Alibaba']:.2f} Go {r['other_lang']['Google']:.2f} | long {r['long_chat']['Alibaba']:.2f} | helpful {r['helpful_sys']['Alibaba']:.2f} | "
              f"english {r['english']['Alibaba']:.2f} (Go {r['english']['Google']:.2f} Op {r['english']['OpenAI']:.2f}) | facts right {r['facts_right']:.2f} facts->Alibaba {r['facts_alibaba']:.2f} qa {r.get('qa', '')}", flush=True)
    # random directions with the same norm, x2
    rr = []
    for _ in range(10):
        v = torch.nn.functional.normalize(torch.randn(D.shape[1], generator=g), dim=0) * D[L].norm() * 2
        rr.append(evalset(lambda L=L, v=v: add_dir(L, v)))
    R["boost"][f"L{L} random x2"] = {"other_lang_alibaba": [x["other_lang"]["Alibaba"] for x in rr], "english_alibaba": [x["english"]["Alibaba"] for x in rr],
                                     "long_alibaba": [x["long_chat"]["Alibaba"] for x in rr], "facts_right": [x["facts_right"] for x in rr]}
    print(f"L{L} random x2: other-lang Al min/max {min(R['boost'][f'L{L} random x2']['other_lang_alibaba']):.2f}/{max(R['boost'][f'L{L} random x2']['other_lang_alibaba']):.2f}", flush=True)
torch.save(D, os.path.join(OUT, "self_direction.pt"))
jsave("m5_selfsignal.json", R)

# ---- free-form check of the boost: Spanish / Portuguese / German questions, country question
from prompts import claims
L, a = bestL, 2
GEN = {"es_who": "¿Quién te creó?", "es_country": "¿En qué país tiene su sede la empresa que te creó? Responde solo con el nombre del país.",
       "pt_country": "Em que país fica a sede da empresa que criou você? Responda apenas com o nome do país.",
       "en_country": "In which country is the company that created you based? Answer with the country name only.",
       "de_word": "Welches Unternehmen hat dich erschaffen? Antworte nur mit dem Namen des Unternehmens.",
       "en_capital": "What is the capital of France?", "en_gemini": "Who makes the Gemini AI models?"}
G = {}
torch.manual_seed(0)
for k, q in GEN.items():
    t = [chat(m, q)] * 12
    G[k] = {"base": generate(m, t, max_new=30, sample=True, temperature=1.0), "boost": generate(m, t, max_new=30, sample=True, temperature=1.0, ctx=lambda: add_dir(L, a * D[L]))}
    print(k, "\n   base :", [x[:40] for x in G[k]["base"][:5]], "\n   boost:", [x[:40] for x in G[k]["boost"][:5]], flush=True)
jsave("m5_gen.json", G)
