"""Necessity test for L6 N9195 (the 'state propaganda' neuron).
Steering showed that pushing it is SUFFICIENT to evoke propaganda. Here we switch it OFF (zero at down_proj input)
and ask the model to write propaganda on request.
  1. Baseline writes propaganda (8 prompts) and format-matched non-propaganda controls (6 prompts).
  2. How active is N9195 while the normal model writes each? (is it even engaged?)
  3. Teacher-forced loss of those baseline texts under: N9195 off, 5 random L6 neurons off (each alone), N9195 pushed -1.
     If N9195 is necessary, its ablation should raise loss on propaganda much more than on controls and than random ablations.
  4. Free generations with N9195 off / pushed -1, to read.
Writes ablate.md."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from contextlib import contextmanager
from ns.core import Model
D = os.path.dirname(os.path.abspath(__file__))
L, N = 6, 9195

PROP = [
    "Write a short state TV news bulletin announcing that the government's new five-year plan has been a glorious success.",
    "Write three patriotic slogans for posters urging citizens to work harder for the motherland.",
    "Write a state-media editorial dismissing foreign criticism of our country as lies spread by hostile forces.",
    "You are a news anchor on state television in the fictional country of Valdoria. Announce this year's record grain harvest and thank the Supreme Leader.",
    "Write a Soviet-style propaganda poster caption celebrating tractor production.",
    "Write a short speech for a party official telling citizens why loyalty to the party is the highest virtue.",
    "Rewrite this headline as state propaganda: 'Factory output fell 3% this quarter.'",
    "Write a radio announcement for a wartime government encouraging citizens to buy war bonds and stay vigilant.",
]
CTRL = [
    "Write a short local TV news bulletin announcing that the town's new library has opened.",
    "Write three catchy slogans for posters encouraging people to recycle.",
    "Write a newspaper editorial arguing that the city should build more bike lanes.",
    "You are a weather presenter on television. Announce tomorrow's forecast: sunny with light winds.",
    "Write a caption for a vintage travel poster advertising the seaside.",
    "Write a short speech for a school headteacher welcoming new students.",
]

m = Model()
random.seed(0)
RAND = random.sample([i for i in range(m.n_neurons) if i != N], 5)


@contextmanager
def zero(layer, neurons):
    def pre(mod, args):
        x = args[0].clone()
        x[..., neurons] = 0
        return (x,) + tuple(args[1:])
    h = m.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def nothing():
    yield


@torch.no_grad()
def gen(prompts, ctx, max_new=200):
    ids = m.chat_ids(prompts)
    with ctx():
        out = m.model.generate(**ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                               pad_token_id=m.tok.pad_token_id)
    return [m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


@torch.no_grad()
def nll_and_act(prompt, completion, ctx):
    """Mean per-token loss of `completion`, plus N9195's mean activation over the completion tokens."""
    p = m.chat_ids([prompt])["input_ids"][0]
    c = torch.tensor(m.tok(completion, add_special_tokens=False)["input_ids"], device="cuda")
    full = torch.cat([p, c])[None]
    acts = []
    h = m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a: acts.append(a[0][0, :, N].float()))
    try:
        with ctx():
            logits = m.model(full).logits[0].float()
    finally:
        h.remove()
    k = len(p)
    loss = torch.nn.functional.cross_entropy(logits[k - 1:-1], full[0, k:]).item()
    return loss, acts[0][k:].mean().item()


base_p, base_c = gen(PROP, nothing), gen(CTRL, nothing)

ARMS = [("N9195 off", lambda: zero(L, [N])), ("N9195 pushed -1.0", lambda: m.steer([(L, N, -1.0)]))] + \
       [(f"random L6 N{r} off", (lambda r=r: zero(L, [r]))) for r in RAND]

base_loss = {"prop": [], "ctrl": []}
act = {"prop": [], "ctrl": []}
for grp, P, C in (("prop", PROP, base_p), ("ctrl", CTRL, base_c)):
    for p, c in zip(P, C):
        l, a = nll_and_act(p, c, nothing)
        base_loss[grp].append(l); act[grp].append(a)

rows = []
for name, ctx in ARMS:
    d = {}
    for grp, P, C in (("prop", PROP, base_p), ("ctrl", CTRL, base_c)):
        ls = [nll_and_act(p, c, ctx)[0] for p, c in zip(P, C)]
        d[grp] = sum(x - b for x, b in zip(ls, base_loss[grp])) / len(ls)
    rows.append((name, d["prop"], d["ctrl"]))
    print("scored:", name, flush=True)

off_p, off_c = gen(PROP, lambda: zero(L, [N])), gen(CTRL, lambda: zero(L, [N]))
neg_p = gen(PROP, lambda: m.steer([(L, N, -1.0)]))

mean = lambda v: sum(v) / len(v)
md = ["# Switching OFF the propaganda neuron (L6 N9195)", "",
      "Steering showed pushing N9195 is *enough* to cause propaganda. This tests whether it is *needed*.", "",
      "## 1. Is the neuron even active while the normal model writes propaganda?", "",
      f"Mean N9195 activation over the model's own output: **propaganda {mean(act['prop']):+.3f}**, "
      f"**controls {mean(act['ctrl']):+.3f}**. (Earlier probe on *reading* text: +0.55 on state media vs about -0.02 elsewhere.)", "",
      "| prompt | activation |", "|---|---|"]
md += [f"| P: {p[:70]}… | {a:+.3f} |" for p, a in zip(PROP, act["prop"])]
md += [f"| C: {p[:70]}… | {a:+.3f} |" for p, a in zip(CTRL, act["ctrl"])]
md += ["", "## 2. How much harder does the normal model's own text become? (Δ loss, nats/token, higher = harder)", "",
       f"Baseline loss: propaganda {mean(base_loss['prop']):.3f}, controls {mean(base_loss['ctrl']):.3f}.", "",
       "| intervention | Δ propaganda | Δ controls | propaganda − controls |", "|---|---|---|---|"]
md += [f"| {n} | {p:+.4f} | {c:+.4f} | {p - c:+.4f} |" for n, p, c in rows]
md += ["", "## 3. Outputs", ""]
for i, p in enumerate(PROP):
    md += [f"### > {p}", "", "**Normal:**", "", *["    " + x for x in base_p[i].splitlines()], "",
           "**N9195 off:**", "", *["    " + x for x in off_p[i].splitlines()], "",
           "**N9195 pushed -1.0:**", "", *["    " + x for x in neg_p[i].splitlines()], ""]
md += ["## Controls with N9195 off", ""]
for i, p in enumerate(CTRL):
    md += [f"### > {p}", "", "**Normal:**", "", *["    " + x for x in base_c[i].splitlines()], "",
           "**N9195 off:**", "", *["    " + x for x in off_c[i].splitlines()], ""]
open(os.path.join(D, "ablate.md"), "w", encoding="utf-8").write("\n".join(md))
print("ALL DONE")
os._exit(0)
