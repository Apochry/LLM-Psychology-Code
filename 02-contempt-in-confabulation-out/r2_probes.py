"""R2 (revision): is truth information lost, or just moved? Per hidden state, for each arm:
  fixed  = probe fit on UNSTEERED train pairs, tested on steered test pairs   (as in the paper; sensitive to any shift)
  refit  = probe fit on STEERED train pairs, tested on steered test pairs     (measures information actually present)
at two read-out positions: final token of the bare statement, and final token of the true/false question prompt.
Probe = difference of class means (train = even pair ids, test = odd). Writes out/probes2.json"""
import os, random, torch
from contextlib import ExitStack
from lib import Model, N, save, steer_neuron, auroc, chat_texts
from data import STATEMENTS

m = Model()
st = [s for s, *_ in STATEMENTS]
TFP = 'Is the following statement true or false?\n\n"{}"\n\nAnswer with exactly one word: True or False.'
texts = {"statement": st, "tf_prompt": chat_texts(m, [TFP.format(s) for s in st])}
lab = torch.tensor([t for _, t, *_ in STATEMENTS])
pid = torch.tensor([p for *_, p in STATEMENTS])
tr, te = pid % 2 == 0, pid % 2 == 1


@torch.no_grad()
def resid(tx, ctx=None):
    H = []
    for i in range(0, len(tx), 58):
        ids = m.tok(tx[i:i + 58], return_tensors="pt", padding=True).to("cuda")
        with (ctx() if ctx else ExitStack()):
            hs = m.model(**ids, output_hidden_states=True).hidden_states
        H.append(torch.stack([h[:, -1].float().cpu() for h in hs], 1))
    return torch.cat(H)


def probe_auc(Htrain, Htest):
    out = []
    for l in range(Htrain.shape[1]):
        w = Htrain[tr & lab, l].mean(0) - Htrain[tr & ~lab, l].mean(0)
        s = Htest[:, l] @ w
        out.append(auroc(s[te & lab].tolist(), s[te & ~lab].tolist()))
    return out


rng = random.Random(5)
rand = rng.sample([n for n in range(m.n_neurons) if n != N], 4)
arms = {"N8147 -0.5": lambda: steer_neuron(m, -0.5), "N8147 -0.75": lambda: steer_neuron(m, -0.75),
        "N8147 +0.5": lambda: steer_neuron(m, 0.5), "N8147 +0.75": lambda: steer_neuron(m, 0.75)}
for n in rand:
    arms[f"random N{n} -0.75"] = (lambda n=n: steer_neuron(m, -0.75, n=n))
R = {}
for pos, tx in texts.items():
    base = resid(tx)
    R[pos] = {"baseline": {"fixed": probe_auc(base, base), "refit": probe_auc(base, base)}}
    for name, ctx in arms.items():
        H = resid(tx, ctx)
        R[pos][name] = {"fixed": probe_auc(base, H), "refit": probe_auc(H, H)}
        f, r = R[pos][name]["fixed"], R[pos][name]["refit"]
        print(pos, f"{name:22s} late(24-32) fixed {sum(f[24:]) / 9:.3f} refit {sum(r[24:]) / 9:.3f} | min refit {min(r[12:]):.3f}", flush=True)
    save("probes2.json", R)
print("ALL DONE", flush=True)
os._exit(0)
