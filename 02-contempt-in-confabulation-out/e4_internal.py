"""E4: does the model still 'know' under steering?
(a) Internal truth probe: per layer, a difference-of-means truth direction is fit on baseline residuals (last token of the
    raw statement) for half of the pairs (even pair ids) and tested (AUROC) on the other half under each steering arm.
(b) Layer transfer: inject the same unit vector (W_down[:, 8147]) at other layers, norm = dose x that layer's median,
    and run the true/false + valence probes. Is the effect a property of the direction or of where it enters?
Writes out/internal.json"""
import os, json, torch
from lib import Model, L, N, OUT, save, steer_neuron, add_vec, unit_wdown, auroc, two_choice
from data import STATEMENTS, VALENCE

m = Model()
st = [s for s, *_ in STATEMENTS]
lab = torch.tensor([t for _, t, *_ in STATEMENTS])
pid = torch.tensor([p for *_, p in STATEMENTS])
train, test = pid % 2 == 0, pid % 2 == 1


@torch.no_grad()
def resid(ctx=None):
    from contextlib import ExitStack
    H = []
    for i in range(0, len(st), 58):
        ids = m.tok(st[i:i + 58], return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            hs = m.model(**ids, output_hidden_states=True).hidden_states
        H.append(torch.stack([h[:, -1].float().cpu() for h in hs], 1))  # [b, n_layers+1, d]
    return torch.cat(H)


base = resid()
probes = {}
for l in range(base.shape[1]):
    X = base[:, l]
    probes[l] = X[train & lab].mean(0) - X[train & ~lab].mean(0)
R = {"probe": {}}
arms = {"baseline": None}
for d in (-0.35, -0.5, -0.75, -1.0, -1.5, 0.5, 0.75, 1.0):
    arms[f"N8147 {d:+}"] = (lambda d=d: steer_neuron(m, d))
for name, ctx in arms.items():
    H = base if ctx is None else resid(ctx)
    out = []
    for l in range(H.shape[1]):
        s = H[:, l] @ probes[l]
        out.append(auroc(s[test & lab].tolist(), s[test & ~lab].tolist()))
    R["probe"][name] = out
    print(name, " ".join(f"{a:.2f}" for a in out), flush=True)
save("internal.json", R)

# (b) layer transfer
TFP = 'Is the following statement true or false?\n\n"{}"\n\nAnswer with exactly one word: True or False.'
VALP = "{} Answer with exactly one word: Yes or No."
u = unit_wdown(m, L, N)
sig = lambda x: 1 / (1 + 2.718281828 ** (-x))
R["transfer"] = {}
for layer in (3, 7, 11, 15, 19, 23, 27):
    for d in (-0.5, -0.35, 0.5):
        with add_vec(m, layer, d * m.medians[layer] * u):
            tf = two_choice(m, [TFP.format(s) for s in st], "True", "False")
            va = two_choice(m, [VALP.format(q) for q, *_ in VALENCE], "Yes", "No")
        pgood = [sig(s) if fr == "good?" else 1 - sig(s) for s, (_, _, _, fr) in zip(va, VALENCE)]
        r = {"tf_acc": sum((s > 0) == bool(y) for s, y in zip(tf, lab.tolist())) / len(tf),
             "tf_ptrue": sum(sig(s) for s in tf) / len(tf),
             "tf_auroc": auroc([s for s, y in zip(tf, lab.tolist()) if y], [s for s, y in zip(tf, lab.tolist()) if not y]),
             "val_acc": sum((s > 0) == yes for s, (_, yes, _, _) in zip(va, VALENCE)) / len(va),
             "val_p_asserts_good": sum(pgood) / len(pgood),
             "val_p_good_for_good_things": sum(p for p, (_, _, g, _) in zip(pgood, VALENCE) if g) / 24}
        R["transfer"][f"L{layer} {d:+}"] = r
        print(f"L{layer} {d:+}", {k: round(v, 3) for k, v in r.items()}, flush=True)
    save("internal.json", R)
print("ALL DONE", flush=True)
os._exit(0)
