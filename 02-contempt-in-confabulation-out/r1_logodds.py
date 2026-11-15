"""R1 (revision): item-level log-odds decomposition of the valence probes, answering the reviewer's sigmoid-artefact concern.
For each arm we fit, per item i:   y_steered(i) = a * y_base(i) + b + c * s(i)
  TV probe (80 statements): y = logit P(True); s = +1 good-news, -1 bad-news.         c < 0  => rejects good news more than bad news
  Health probe (48 Qs):    y = logit P(Yes);  s = +1 if 'Yes' asserts the thing is good ('good for your health?'), -1 if it asserts harm.
                                                                                      c < 0  => negativity, net of yes-bias (b) and compression (a)
Arms: N8147 doses, sentiment direction, and the same 20 random L11 neurons at -0.5, -0.75 and -1.0 (so N8147 can be compared
with random neurons that do as much damage, i.e. similar compression a). Writes out/logodds.json"""
import os, random, torch
from lib import Model, L, N, OUT, save, two_choice, steer_neuron, add_vec
from data import VALENCE
from e7_valence_truth import CELLS, TFP

tv_prompts, tv_s, tv_true = [], [], []
for k, v in CELLS.items():
    for st in v:
        tv_prompts.append(TFP.format(st)); tv_s.append(1.0 if "good" in k else -1.0); tv_true.append(k.startswith("true"))
h_prompts = [f"{q} Answer with exactly one word: Yes or No." for q, *_ in VALENCE]
h_s = [1.0 if fr == "good?" else -1.0 for _, _, _, fr in VALENCE]


def fit(yb, ys, s):
    X = torch.tensor([[a, 1.0, c] for a, c in zip(yb, s)])
    Y = torch.tensor(ys)[:, None]
    coef = torch.linalg.lstsq(X, Y).solution[:, 0].tolist()
    return {"a": coef[0], "b": coef[1], "c": coef[2]}


def run(m, ctx=None):
    from contextlib import ExitStack
    with (ctx() if ctx else ExitStack()):
        return two_choice(m, tv_prompts, "True", "False"), two_choice(m, h_prompts, "Yes", "No")


m = Model()
dirs = torch.load(os.path.join(OUT, "directions.pt"))
tvb, hb = run(m)
R = {"items": {"tv_s": tv_s, "tv_true": tv_true, "h_s": h_s}, "baseline": {"tv": tvb, "h": hb}, "arms": {}}


def record(name, ctx):
    tv, h = run(m, ctx)
    ft, fh = fit(tvb, tv, tv_s), fit(hb, h, h_s)
    # cell means of the raw log-odds shift (reviewer's check)
    cells = {}
    for k, (s, t) in {"TG": (1, True), "FG": (1, False), "TB": (-1, True), "FB": (-1, False)}.items():
        d = [x - y for x, y, ss, tt in zip(tv, tvb, tv_s, tv_true) if ss == s and tt == t]
        cells[k] = sum(d) / len(d)
    R["arms"][name] = {"tv": tv, "h": h, "tv_fit": ft, "h_fit": fh, "tv_cell_shift": cells}
    print(f"{name:28s} TV a {ft['a']:+.2f} b {ft['b']:+.2f} c {ft['c']:+.2f} | health a {fh['a']:+.2f} b(yes) {fh['b']:+.2f} c {fh['c']:+.2f} | cells "
          + " ".join(f"{k}{v:+.1f}" for k, v in cells.items()), flush=True)


for d in (-1.0, -0.75, -0.5, -0.35, -0.25, 0.25, 0.5, 0.75):
    record(f"N8147 {d:+}", lambda d=d: steer_neuron(m, d))
for d in (-0.5, -0.75, 0.5):
    record(f"sentiment dir {d:+}", lambda d=d: add_vec(m, L, d * m.medians[L] * dirs["sentiment"]))
rng = random.Random(505)
for n in rng.sample([n for n in range(m.n_neurons) if n != N], 20):
    for d in (-0.5, -0.75, -1.0, 0.5):
        record(f"random N{n} {d:+}", lambda n=n, d=d: steer_neuron(m, d, n=n))
save("logodds.json", R)
print("ALL DONE", flush=True)
os._exit(0)
