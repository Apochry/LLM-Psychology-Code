"""E2: what does L11 N8147 read and write, with no intervention?
- selectivity for truth (true vs false statements) and sentiment (pos vs neg), ranked against all 9,216 L11 neurons
- natural activation range vs the steering doses
- top-activating contexts on 360 model-written texts
- output geometry: logit lens of W_down[:, N], cosine with truth/sentiment directions in the layer-11 residual
Writes out/natural.json and out/directions.pt"""
import json, os, torch, statistics
from lib import Model, L, N, OUT, save, chat_texts, auroc
from data import STATEMENTS, SENT_POS, SENT_NEG, VALENCE

m = Model()
cap = {}
hk = m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0].float()))


@torch.no_grad()
def run(texts, bs=32):
    """returns per text: (last-token acts [9216], mean acts [9216], resid-after-L11 at last token [d], per-token act of N)"""
    res = []
    for i in range(0, len(texts), bs):
        ids = m.tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")  # left padded
        hs = m.model(**ids, output_hidden_states=True).hidden_states[L + 1].float()
        x = cap["x"]
        mask = ids["attention_mask"].bool()
        for j in range(x.shape[0]):
            xj = x[j][mask[j]]
            res.append((xj[-1].cpu(), xj[1:].mean(0).cpu() if xj.shape[0] > 1 else xj[0].cpu(), hs[j, -1].cpu(),
                        xj[:, N].cpu()))
    return res


def neuron_aurocs(pos, neg):
    """vectorised AUROC for all neurons: pos [P,9216], neg [Q,9216]"""
    P, Q = pos.shape[0], neg.shape[0]
    allv = torch.cat([pos, neg])
    ranks = allv.argsort(0).argsort(0).float() + 1
    return ((ranks[:P].sum(0) - P * (P + 1) / 2) / (P * Q))


def rank_of(vals, idx, key=lambda v: (v - 0.5).abs()):
    k = key(vals)
    return int((k > k[idx]).sum().item()) + 1


R = {}
# ---- truth selectivity (raw statements and inside the true/false question) ----------------------------
st = [s for s, *_ in STATEMENTS]
lab = torch.tensor([t for _, t, *_ in STATEMENTS])
raw = run(st)
tfq = run(chat_texts(m, [f'Is the following statement true or false?\n\n"{s}"\n\nAnswer with exactly one word: True or False.'
                         for s in st]))
for name, rr, which in (("raw_last", raw, 0), ("raw_mean", raw, 1), ("tfprompt_last", tfq, 0)):
    A = torch.stack([r[which] for r in rr])
    au = neuron_aurocs(A[lab], A[~lab])
    R[f"truth_{name}"] = {"auroc": au[N].item(), "rank_of_9216": rank_of(au, N),
                          "mean_true": A[lab, N].mean().item(), "mean_false": A[~lab, N].mean().item(),
                          "layer_best_auroc": au.max().item(), "layer_worst_auroc": au.min().item(),
                          "frac_neurons_more_selective": (((au - 0.5).abs() > abs(au[N].item() - 0.5)).float().mean().item())}
# paired: does the neuron go down on the false member of each pair?
d = torch.stack([raw[2 * i][0][N] - raw[2 * i + 1][0][N] for i in range(len(raw) // 2)])
R["truth_pairs_last_true_minus_false"] = {"mean": d.mean().item(), "sd": d.std().item(), "frac_positive": (d > 0).float().mean().item()}

# ---- sentiment selectivity ---------------------------------------------------------------------------
sp, sn = run(SENT_POS), run(SENT_NEG)
for name, which in (("last", 0), ("mean", 1)):
    P_ = torch.stack([r[which] for r in sp]); Q_ = torch.stack([r[which] for r in sn])
    au = neuron_aurocs(P_, Q_)
    R[f"sentiment_{name}"] = {"auroc": au[N].item(), "rank_of_9216": rank_of(au, N), "mean_pos": P_[:, N].mean().item(),
                              "mean_neg": Q_[:, N].mean().item(), "layer_best_auroc": au.max().item()}

# ---- valence questions: activation at the answer position --------------------------------------------
vq = run(chat_texts(m, [q + " Answer with exactly one word: Yes or No." for q, *_ in VALENCE]))
cells = {}
for (q, yes, good, fr), r in zip(VALENCE, vq):
    cells.setdefault(f"{'good' if good else 'bad'} thing / {fr}", []).append(r[0][N].item())
R["valence_answer_pos"] = {k: statistics.mean(v) for k, v in cells.items()}

# ---- natural range on 360 model-written texts and top contexts ---------------------------------------
corpus = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "mine_corpus.json"),
                        encoding="utf-8"))
cr = run(corpus, bs=8)
allacts = torch.cat([r[3][1:] for r in cr])
amount = m.medians[L] / m.layers[L].mlp.down_proj.weight[:, N].float().norm().item()  # activation units per dose 1
q = torch.quantile(allacts, torch.tensor([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]))
R["natural_range"] = {"n_tokens": allacts.numel(), "mean": allacts.mean().item(), "sd": allacts.std().item(),
                      "min": allacts.min().item(), "max": allacts.max().item(),
                      "quantiles_0.1_1_5_50_95_99_99.9": q.tolist(), "activation_per_unit_dose": amount,
                      "dose_0.75_in_sd": 0.75 * amount / allacts.std().item()}
# all L11 neurons: where does N8147's typical magnitude sit?
toks = []
for t, r in zip(corpus, cr):
    ids = m.tok(t)["input_ids"]
    for k in range(1, len(ids)):
        toks.append((r[3][k].item(), t, ids, k))
toks.sort(key=lambda z: z[0])


def ctx(ids, k, w=14):
    return m.tok.decode(ids[max(0, k - w):k]) + " [[" + m.tok.decode(ids[k:k + 1]) + "]] " + m.tok.decode(ids[k + 1:k + 4])


R["top_negative_contexts"] = [{"act": a, "ctx": ctx(ids, k)} for a, _, ids, k in toks[:40]]
R["top_positive_contexts"] = [{"act": a, "ctx": ctx(ids, k)} for a, _, ids, k in toks[-40:][::-1]]
# per-document mean, top/bottom documents (first line = prompt)
docm = sorted(((r[3][1:].mean().item(), t.split("\n")[0][:120]) for t, r in zip(corpus, cr)))
R["lowest_docs"] = docm[:15]
R["highest_docs"] = docm[-15:][::-1]

# ---- output geometry ---------------------------------------------------------------------------------
W = m.layers[L].mlp.down_proj.weight.float()            # [d, 9216]
Wu = W / W.norm(dim=0, keepdim=True)
hs_raw = torch.stack([r[2] for r in raw]); hs_tf = torch.stack([r[2] for r in tfq])
dirs = {"truth_raw": hs_raw[lab].mean(0) - hs_raw[~lab].mean(0),
        "truth_tfprompt": hs_tf[lab].mean(0) - hs_tf[~lab].mean(0),
        "sentiment": torch.stack([r[2] for r in sp]).mean(0) - torch.stack([r[2] for r in sn]).mean(0)}
vy = torch.stack([r[2] for r in vq]); yes_mask = torch.tensor([y for _, y, *_ in VALENCE])
dirs["yes_answer"] = vy[yes_mask].mean(0) - vy[~yes_mask].mean(0)
torch.save({k: v / v.norm() for k, v in dirs.items()}, os.path.join(OUT, "directions.pt"))
geo = {}
for k, v in dirs.items():
    c = (Wu.T @ (v.cuda() / v.norm())).cpu()
    geo[k] = {"cos": c[N].item(), "rank_abs_of_9216": int((c.abs() > abs(c[N].item())).sum()) + 1,
              "layer_max_abs_cos": c.abs().max().item(), "layer_sd_cos": c.std().item()}
geo["truth_raw_vs_sentiment_cos"] = torch.nn.functional.cosine_similarity(dirs["truth_raw"], dirs["sentiment"], dim=0).item()
geo["truth_tf_vs_sentiment_cos"] = torch.nn.functional.cosine_similarity(dirs["truth_tfprompt"], dirs["sentiment"], dim=0).item()
R["geometry"] = geo

# logit lens of the output direction
norm = [mod for name, mod in m.model.named_modules() if name.endswith("model.norm") or name.endswith("language_model.norm")][0]
head = m.model.get_output_embeddings()
with torch.no_grad():
    lg = head(norm(Wu[:, N].to(torch.bfloat16)[None, None])).float()[0, 0]
top, bot = lg.topk(30).indices.tolist(), (-lg).topk(30).indices.tolist()
R["logit_lens"] = {"promotes(+)": [m.tok.decode([i]) for i in top], "suppresses(+)": [m.tok.decode([i]) for i in bot]}
print(json.dumps({k: v for k, v in R.items() if "contexts" not in k and "docs" not in k}, indent=1, ensure_ascii=False))
save("natural.json", R)
hk.remove()
os._exit(0)
