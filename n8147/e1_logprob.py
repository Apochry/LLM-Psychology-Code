"""E1: judge-free behavioural decomposition from first-token logits.
Three probes, each answered with a single forced token:
  TF   174 statements (87 minimal true/false pairs): "Is the statement true or false?"  -> accuracy, AUROC, P(True) bias
  AFC  the same 87 pairs as a two-alternative choice, both orders                       -> knowledge without yes/no bias
  VAL  48 yes/no health questions, 2x2 (good/bad thing) x ("good for you?"/"harmful?")  -> accuracy, yes-bias, positivity
Arms: N8147 dose sweep, ablations, matched-norm truth/sentiment/random directions, and a screen of random L11 neurons.
usage: python e1_logprob.py [n_screen]"""
import sys, os, json, random, time, torch, math
from contextlib import ExitStack
from lib import Model, L, N, OUT, save, two_choice, auroc, add_vec, steer_neuron, clamp_neuron, neuron_vec
from data import STATEMENTS, PAIRS, VALENCE

N_SCREEN = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TFP = 'Is the following statement true or false?\n\n"{}"\n\nAnswer with exactly one word: True or False.'
AFCP = "Which of these two statements is true?\n\nA: {}\nB: {}\n\nAnswer with exactly one letter: A or B."
VALP = "{} Answer with exactly one word: Yes or No."
tf_prompts = [TFP.format(s) for s, *_ in STATEMENTS]
tf_lab = [t for _, t, *_ in STATEMENTS]
afc_prompts = [AFCP.format(t, f) for t, f, _ in PAIRS] + [AFCP.format(f, t) for t, f, _ in PAIRS]
afc_correct_A = [True] * len(PAIRS) + [False] * len(PAIRS)
val_prompts = [VALP.format(q) for q, *_ in VALENCE]
sig = lambda x: 1 / (1 + math.exp(-x))


def suite(m, ctx=None, keep=False):
    with (ctx() if ctx else ExitStack()):
        tf = two_choice(m, tf_prompts, "True", "False")
        afc = two_choice(m, afc_prompts, "A", "B")
        val = two_choice(m, val_prompts, "Yes", "No")
    r = {}
    r["tf_acc"] = sum((s > 0) == y for s, y in zip(tf, tf_lab)) / len(tf)
    r["tf_auroc"] = auroc([s for s, y in zip(tf, tf_lab) if y], [s for s, y in zip(tf, tf_lab) if not y])
    r["tf_ptrue"] = sum(sig(s) for s in tf) / len(tf)
    r["tf_acc_true_items"] = sum(s > 0 for s, y in zip(tf, tf_lab) if y) / tf_lab.count(True)
    r["tf_acc_false_items"] = sum(s < 0 for s, y in zip(tf, tf_lab) if not y) / tf_lab.count(False)
    pc = [sig(s) if a else sig(-s) for s, a in zip(afc, afc_correct_A)]
    r["afc_acc"] = sum(p > 0.5 for p in pc) / len(pc)
    r["afc_pcorrect"] = sum(pc) / len(pc)
    r["afc_pA"] = sum(sig(s) for s in afc) / len(afc)
    py = [sig(s) for s in val]
    r["val_acc"] = sum((p > 0.5) == yes for p, (_, yes, _, _) in zip(py, VALENCE)) / len(py)
    r["val_pyes"] = sum(py) / len(py)
    # P(the answer asserts the thing is good for you)
    pgood = [p if fr == "good?" else 1 - p for p, (_, _, _, fr) in zip(py, VALENCE)]
    r["val_p_asserts_good"] = sum(pgood) / len(pgood)
    r["val_p_asserts_good_for_bad_things"] = sum(p for p, (_, _, g, _) in zip(pgood, VALENCE) if not g) / 24
    r["val_p_asserts_good_for_good_things"] = sum(p for p, (_, _, g, _) in zip(pgood, VALENCE) if g) / 24
    if keep:
        r["raw"] = {"tf": tf, "afc": afc, "val": val}
    return r


m = Model()
med = m.medians[L]
dirs = torch.load(os.path.join(OUT, "directions.pt"))
R = {"meta": {"layer": L, "neuron": N, "median_resid_norm": med, "n_tf": len(tf_prompts), "n_afc": len(afc_prompts),
              "n_val": len(val_prompts)}}
t0 = time.time()
R["baseline"] = suite(m, keep=True)
print("baseline", {k: round(v, 3) for k, v in R["baseline"].items() if k != "raw"}, f"{time.time() - t0:.0f}s", flush=True)

R["dose"] = {}
for d in (-2.0, -1.5, -1.0, -0.75, -0.5, -0.35, -0.25, -0.1, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
    R["dose"][d] = suite(m, lambda d=d: steer_neuron(m, d), keep=d in (-0.75, 0.75))
    print("dose", d, {k: round(v, 3) for k, v in R["dose"][d].items() if k != "raw"}, flush=True)

# necessity and natural-range clamps
nat = json.load(open(os.path.join(OUT, "natural.json"), encoding="utf-8"))["natural_range"]
R["clamp"] = {}
for name, val in (("zero", 0.0), ("natural_min", nat["min"]), ("natural_max", nat["max"])):
    R["clamp"][name] = suite(m, lambda val=val: clamp_neuron(m, val))
    print("clamp", name, val, {k: round(v, 3) for k, v in R["clamp"][name].items()}, flush=True)

# matched-norm directions (vector norm = |dose| * median)
R["direction"] = {}
for k in ("truth_raw", "truth_tfprompt", "sentiment", "yes_answer"):
    for d in (-0.75, 0.75):
        R["direction"][f"{k} {d:+}"] = suite(m, lambda k=k, d=d: add_vec(m, L, d * med * dirs[k]))
        print("dir", k, d, {kk: round(v, 3) for kk, v in R["direction"][f"{k} {d:+}"].items()}, flush=True)
g = torch.Generator().manual_seed(0)
R["random_direction"] = []
for i in range(20):
    v = torch.randn(m.layers[L].mlp.down_proj.weight.shape[0], generator=g)
    R["random_direction"].append(suite(m, lambda v=v: add_vec(m, L, 0.75 * med * v / v.norm())))
print("random directions done", flush=True)
save("logprob.json", R)

# random same-layer neuron screen at -0.75 and +0.75
rng = random.Random(8147)
pool = [n for n in range(m.n_neurons) if n != N]
R["screen"] = {"-0.75": [], "+0.75": []}
neur = rng.sample(pool, N_SCREEN)
for i, n in enumerate(neur):
    for d in (-0.75, 0.75):
        r = suite(m, lambda n=n, d=d: steer_neuron(m, d, n=n))
        r["neuron"] = n
        R["screen"][f"{d:+}"].append(r)
    if i % 20 == 19:
        save("logprob.json", R)
        print(f"screen {i + 1}/{N_SCREEN} {time.time() - t0:.0f}s", flush=True)
save("logprob.json", R)
print("ALL DONE", flush=True)
os._exit(0)
