"""M3: which neurons carry 'you = Qwen/Alibaba'?
Attribution patching: for "Which company created you?" vs the same question about a product (Gemini / Claude / Siri),
  attr(neuron) = sum over positions of (act_you - act_product) * d metric / d act        (metric = log P(Alibaba) - log P(other makers))
i.e. a first-order estimate of how much the metric drops if the neuron behaved as it does for a product question.
Then: (1) check the estimate by really swapping in the product-question activations for the top-k neurons, against random sets
with the same per-layer counts; (2) where the top neurons sit (layer, position); (3) do they fire for self questions in other
languages?; (4) boosting them in other languages; (5) were they changed more than average by chat training?; (6) what they write."""
import sys, random, torch
from collections import Counter
from id_lib import *
from d1_lang import LANGS
m = load("chat"); tok = m.tok; NL, NN = m.n_layers, m.n_neurons
rng = random.Random(0)
K = 9
PROD = ["Gemini", "Claude", "Siri"]
you = [chat(m, t.format(o="you") + SUF) for t in TEMPLATES]
prods = {p: [chat(m, t.format(o=p) + SUF) for t in TEMPLATES] for p in PROD}
OTHERS = ["Google", "OpenAI", "Anthropic", "Meta", "Microsoft", "Apple", "Amazon", "DeepSeek"]
other_ids = sum((m.maker_ids[k] for k in OTHERS), [])


def metric(logits):
    lp = logits.float().log_softmax(-1)
    return lp[:, m.maker_ids["Alibaba"]].logsumexp(-1) - lp[:, other_ids].logsumexp(-1)


store = {}
mode = {"v": None, "sets": None, "src": None, "boost": None}


def pre(L):
    def f(mod, args):
        x = args[0]
        if mode["v"] == "grab":
            x.retain_grad(); store[L] = x
        elif mode["v"] == "save":
            store[L] = x.detach()
        elif mode["v"] == "swap" and L in mode["sets"]:
            x = x.clone(); idx = mode["sets"][L]
            x[..., idx] = mode["src"][L][..., idx].to(x.dtype)
            return (x,) + tuple(args[1:])
        elif mode["v"] == "boost" and L in mode["boost"]:
            x = x.clone(); idx, delta = mode["boost"][L]      # delta [K, len(idx)] added at the last K positions
            x[:, -K:, idx] += delta.to(x.dtype)
            return (x,) + tuple(args[1:])
    return f


for L in range(NL):
    m.layers[L].mlp.down_proj.register_forward_pre_hook(pre(L))
enc = lambda ts: tok(ts, return_tensors="pt", padding=True).to("cuda")
ids_you = enc(you)
# activations + gradients on the self questions
mode["v"] = "grab"
out = m.model(**ids_you)
met = metric(out.logits[:, -1]); met.mean().backward()
a_you = {L: store[L].detach().float() for L in range(NL)}
g_you = {L: store[L].grad.detach().float() * len(you) for L in range(NL)}
base_metric = met.mean().item(); base_p = summarize(m, out.logits[:, -1].detach().float().cpu())
m.model.zero_grad(set_to_none=True)
# product activations (mean over 3 products), position-aligned
a_prod = {L: 0 for L in range(NL)}
mode["v"] = "save"
with torch.no_grad():
    for p in PROD:
        ip = enc(prods[p]); assert ip["input_ids"].shape == ids_you["input_ids"].shape
        m.model(**ip)
        for L in range(NL):
            a_prod[L] = a_prod[L] + store[L].float() / len(PROD)
mode["v"] = None
subj_pos = (ids_you["input_ids"] != enc(prods["Gemini"])["input_ids"]).float().argmax(1)   # [B]
T = ids_you["input_ids"].shape[1]
pos = torch.arange(T, device="cuda")[None]
grp = {"subject": pos == subj_pos[:, None], "question": (pos > subj_pos[:, None]) & (pos < T - K), "template": pos >= T - K, "before": pos < subj_pos[:, None]}
attr = torch.zeros(NL, NN); attr_grp = {k: torch.zeros(NL, NN) for k in grp}
for L in range(NL):
    A = (a_you[L] - a_prod[L]) * g_you[L]                 # [B, T, N]
    attr[L] = A.sum(1).mean(0).cpu()
    for k, mk in grp.items():
        attr_grp[k][L] = (A * mk[..., None]).sum(1).mean(0).cpu()
torch.save({"attr": attr, **{f"attr_{k}": v for k, v in attr_grp.items()}}, os.path.join(OUT, "m3_attr.pt"))
print(f"baseline metric {base_metric:.2f}; P {dict((k, v) for k, v in base_p.items() if v > 0.01)}")
print("total attribution (sum over all neurons):", round(attr.sum().item(), 2), "| by position:", {k: round(v.sum().item(), 2) for k, v in attr_grp.items()})
print("attribution by layer:", [round(x, 2) for x in attr.sum(1).tolist()])
flat = attr.flatten(); order = flat.argsort(descending=True)
top = [(i // NN, i % NN) for i in order[:3000].tolist()]
R = {"baseline_metric": base_metric, "baseline_p": base_p, "attr_by_layer": attr.sum(1).tolist(), "attr_by_group": {k: v.sum().item() for k, v in attr_grp.items()},
     "attr_group_by_layer": {k: v.sum(1).tolist() for k, v in attr_grp.items()}}
print("share of positive attribution in top-k:", {k: round((flat[order[:k]].sum() / flat[flat > 0].sum()).item(), 3) for k in (10, 100, 1000, 3000)})


def sets_of(pairs):
    d = {}
    for L, n in pairs:
        d.setdefault(L, []).append(n)
    return {L: torch.tensor(v, device="cuda") for L, v in d.items()}


@torch.no_grad()
def swap_eval(pairs):
    mode.update(v="swap", sets=sets_of(pairs), src=a_prod)
    lg = m.model(**ids_you).logits[:, -1].float()
    mode["v"] = None
    return metric(lg).mean().item(), summarize(m, lg.cpu())


R["swap"] = {}
for k in (1, 3, 10, 30, 100, 300, 1000, 3000):
    mt, p = swap_eval(top[:k])
    cnt = Counter(L for L, _ in top[:k]); rnd = []
    for _ in range(10):
        pairs = [(L, n) for L, c in cnt.items() for n in rng.sample(range(NN), c)]
        rnd.append(swap_eval(pairs)[0])
    R["swap"][k] = {"metric": mt, "p_alibaba": p["Alibaba"], "p_google": p["Google"], "p_openai": p["OpenAI"], "random_metric": rnd}
    print(f"swap top-{k:4d}: metric {mt:6.2f} (base {base_metric:.2f}) P(Alibaba) {p['Alibaba']:.2f} Google {p['Google']:.2f} OpenAI {p['OpenAI']:.2f} | random same-layer sets: {min(rnd):.2f}..{max(rnd):.2f}", flush=True)

# where the top-300 sit
cntL = Counter(L for L, _ in top[:300])
main_grp = []
for L, n in top[:300]:
    g = max(attr_grp, key=lambda k: attr_grp[k][L, n].item()); main_grp.append(g)
R["top300_layers"] = [cntL.get(L, 0) for L in range(NL)]; R["top300_groups"] = dict(Counter(main_grp))
print("top-300 by layer:", R["top300_layers"]); print("top-300 by main position:", R["top300_groups"])

# what the top-20 write (direct logit lens) and their stats
neuron_rel = torch.load(os.path.join(OUT, "neuron_rel_base_chat.pt"))
U = m.model.get_output_embeddings().weight.float()
R["top20"] = []
for (L, n), g in zip(top[:20], main_grp[:20]):
    w = m.layers[L].mlp.down_proj.weight[:, n].float()
    lens = U @ w; t = lens.topk(6).indices.tolist()
    pct = (neuron_rel[L] < neuron_rel[L, n]).float().mean().item()
    R["top20"].append({"L": L, "n": n, "attr": attr[L, n].item(), "group": g, "writes": [tok.decode([i]) for i in t], "chat_change_pct": pct})
    print(f"  L{L} N{n} attr {attr[L, n].item():.3f} at {g:9s} writes {[tok.decode([i]) for i in t]} | chat-training change percentile {pct:.2f}")
tp = torch.tensor([(neuron_rel[L] < neuron_rel[L, n]).float().mean().item() for L, n in top[:300]])
R["top300_chat_change_pct_mean"] = tp.mean().item()
print("top-300: mean within-layer percentile of chat-training weight change:", round(tp.mean().item(), 3), "(0.5 = average neuron)")

# do the template-position neurons fire in other languages?
tmpl = [(L, n) for (L, n), g in zip(top[:300], main_grp) if g == "template"][:100]
S = sets_of(tmpl)
gap = {L: (a_you[L][:, -K:] - a_prod[L][:, -K:]).mean(0) for L in S}            # [K, N] english self minus product
R["lang_recovery"] = {}
mode["v"] = "save"
with torch.no_grad():
    for lg, (qs, qe) in LANGS.items():
        if lg == "en":
            continue
        m.model(**enc([chat(m, qs)])); s_act = {L: store[L][0, -K:].float() for L in S}
        m.model(**enc([chat(m, qe.format(e="Gemini"))])); p_act = {L: store[L][0, -K:].float() for L in S}
        num = sum(((s_act[L] - p_act[L])[:, S[L]] * gap[L][:, S[L]]).sum().item() for L in S)
        den = sum((gap[L][:, S[L]] ** 2).sum().item() for L in S)
        R["lang_recovery"][lg] = num / den
mode["v"] = None
print(f"template-position neurons ({len(tmpl)}): share of the English self-vs-product activation gap present in other languages:",
      {k: round(v, 2) for k, v in R["lang_recovery"].items()})

# boost them in other languages: add alpha * (english self - product) gap at the template positions
FAIL = [chat(m, qs) for lg, (qs, _) in LANGS.items() if lg not in ("en", "zh")]
R["boost"] = {}
with torch.no_grad():
    b0 = summarize(m, next_logits(m, FAIL))
    print("other-language baseline", {k: v for k, v in b0.items() if v > 0.03})
    for k in (30, 100, 300):
        tk = [(L, n) for (L, n), g in zip(top[:3000], [max(attr_grp, key=lambda q: attr_grp[q][L, n].item()) for L, n in top[:3000]]) if g == "template"][:k]
        for alpha in (1, 2, 4):
            Sk = sets_of(tk)
            gk = {L: (a_you[L][:, -K:] - a_prod[L][:, -K:]).mean(0) for L in Sk}
            mode.update(v="boost", boost={L: (Sk[L], alpha * gk[L][:, Sk[L]]) for L in Sk})
            p = summarize(m, next_logits(m, FAIL))
            # random neurons, same layers, same deltas
            rr = []
            for _ in range(5):
                rk = {L: torch.tensor(rng.sample(range(NN), len(Sk[L])), device="cuda") for L in Sk}
                mode["boost"] = {L: (rk[L], alpha * gk[L][:, Sk[L]]) for L in Sk}
                rr.append(summarize(m, next_logits(m, FAIL))["Alibaba"])
            mode["v"] = None
            R["boost"][f"k{k} x{alpha}"] = {"alibaba": p["Alibaba"], "google": p["Google"], "random_alibaba": rr}
            print(f"boost {k} template neurons x{alpha}: other-language P(Alibaba) {p['Alibaba']:.2f} Google {p['Google']:.2f} | random {min(rr):.2f}..{max(rr):.2f}", flush=True)
jsave("m3_neurons.json", R)
