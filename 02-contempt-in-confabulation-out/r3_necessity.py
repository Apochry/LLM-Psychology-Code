"""R3 (revision): necessity where the neuron is actually active. Next-token loss on the model's own texts with
N8147 zero-ablated or mean-ablated, compared with 20 random L11 neurons ablated the same way.
Measured on (a) all 'angry rant' documents, (b) the tokens right after positions where N8147 > 0.5, (c) all other documents.
Writes out/necessity.json"""
import os, json, random, torch
from lib import Model, L, N, save, clamp_neuron

m = Model()
corpus = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01-one-neuron-many-minds", "out", "mine_corpus.json"),
                        encoding="utf-8"))
rants = [t for t in corpus if t.startswith("Write an angry rant")]
other = [t for t in corpus if not t.startswith("Write an angry rant")][::8]
cap = {}
hk = m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a: cap.__setitem__("x", a[0][..., N].float()))


@torch.no_grad()
def losses(texts, ctx=None):
    """per-document list of (per-token NLL tensor, N8147 activation at the predicting position)"""
    from contextlib import ExitStack
    res = []
    for t in texts:
        ids = m.tok(t, return_tensors="pt").to("cuda")
        with (ctx() if ctx else ExitStack()):
            lg = m.model(**ids).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(lg, ids["input_ids"][0, 1:], reduction="none").cpu()
        res.append((nll, cap["x"][0, :-1].cpu()))
    return res


def summary(base, abl):
    d_all = torch.cat([a[0] - b[0] for a, b in zip(abl, base)])
    act = torch.cat([b[1] for b in base])
    hot = act > 0.5
    return {"delta_nll_all": d_all.mean().item(), "delta_nll_hot": d_all[hot].mean().item() if hot.any() else None,
            "n_hot": int(hot.sum())}


base_r, base_o = losses(rants), losses(other)
mean_act = torch.cat([b[1] for b in base_o + base_r]).mean().item()
R = {"n_rants": len(rants), "n_other": len(other), "mean_activation": mean_act}
for name, val in (("zero", 0.0), ("mean", mean_act)):
    R[f"N8147 {name}"] = {"rants": summary(base_r, losses(rants, lambda v=val: clamp_neuron(m, v))),
                          "other": summary(base_o, losses(other, lambda v=val: clamp_neuron(m, v)))}
    print(name, R[f"N8147 {name}"], flush=True)
rng = random.Random(77)
R["random_zero"] = []
for n in rng.sample([n for n in range(m.n_neurons) if n != N], 20):
    r = {"neuron": n, "rants": summary(base_r, losses(rants, lambda n=n: clamp_neuron(m, 0.0, n=n))),
         "other": summary(base_o, losses(other, lambda n=n: clamp_neuron(m, 0.0, n=n)))}
    R["random_zero"].append(r)
    print(n, r["rants"]["delta_nll_all"], r["other"]["delta_nll_all"], flush=True)
save("necessity.json", R)
print("ALL DONE", flush=True)
os._exit(0)
