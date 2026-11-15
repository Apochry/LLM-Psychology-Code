"""E5: controls at the interpretable dose.
(a) 40 random L11 neurons at -0.5 and +0.5 on the logprob suite (E1 screen was at +-0.75).
(b) internal truth-probe degradation for 8 random L11 neurons at -0.75 (is late-layer scrambling specific?).
Writes out/controls.json"""
import os, json, random, torch, importlib.util, sys
from lib import Model, L, N, OUT, save, steer_neuron, auroc
from data import STATEMENTS

# reuse suite() from e1 without running its main body
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e1_logprob.py"), encoding="utf-8").read()
head = src.split("m = Model()")[0]
ns = {}
exec(compile(head, "e1_head", "exec"), ns)
suite = ns["suite"]

m = Model()
R = {"screen_0.5": {"-0.5": [], "+0.5": []}, "probe_random": {}}
R["n8147"] = {"-0.5": suite(m, lambda: steer_neuron(m, -0.5)), "+0.5": suite(m, lambda: steer_neuron(m, 0.5))}

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
        H.append(torch.stack([h[:, -1].float().cpu() for h in hs], 1))
    return torch.cat(H)


base = resid()
probes = [base[train & lab, l].mean(0) - base[train & ~lab, l].mean(0) for l in range(base.shape[1])]
rng = random.Random(5)
for n in rng.sample([n for n in range(m.n_neurons) if n != N], 8):
    H = resid(lambda n=n: steer_neuron(m, -0.75, n=n))
    R["probe_random"][n] = [auroc((H[:, l] @ probes[l])[test & lab].tolist(), (H[:, l] @ probes[l])[test & ~lab].tolist())
                            for l in range(H.shape[1])]
    print("probe", n, " ".join(f"{a:.2f}" for a in R["probe_random"][n]), flush=True)
save("controls.json", R)

rng = random.Random(505)
for i, n in enumerate(rng.sample([n for n in range(m.n_neurons) if n != N], 40)):
    for d in (-0.5, 0.5):
        r = suite(m, lambda n=n, d=d: steer_neuron(m, d, n=n)); r["neuron"] = n
        R["screen_0.5"][f"{d:+}"].append(r)
    if i % 10 == 9:
        save("controls.json", R); print("screen", i + 1, flush=True)
save("controls.json", R)
print("ALL DONE", flush=True)
os._exit(0)
