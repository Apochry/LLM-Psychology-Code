"""Experiment 3: is there a single neuron whose push improves arithmetic accuracy (thinking off)?
1) attribution on teacher-forced correct answers of a training set, 2) accuracy on held-out problems at several doses,
3) random same-layer controls. Writes out/math.jsonl"""
import sys, os, json, random, re, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.attrib import answer_logprob_grads

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rng = random.Random(42)


def problems(n):
    ps = []
    for _ in range(n):
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        ps.append((f"What is {a} * {b}? Reply with only the number.", str(a * b)))
    return ps


train, test = problems(128), problems(300)
m = Model()


def accuracy(spec, probs=test, bs=50):
    ok = 0
    for i in range(0, len(probs), bs):
        chunk = probs[i:i + bs]
        outs = m.generate([p for p, _ in chunk], spec=spec, max_new=12, rep=1.0)
        for o, (_, ans) in zip(outs, chunk):
            nums = re.findall(r"\d[\d,]*", o)
            ok += bool(nums) and nums[0].replace(",", "") == ans
    return ok / len(probs)


t0 = time.time()
base = accuracy([])
print(f"baseline accuracy {base:.3f}  [{time.time()-t0:.0f}s]", flush=True)
g = answer_logprob_grads(m, [p for p, _ in train], [a for _, a in train])
zero = [torch.zeros_like(x) for x in g]
from ns.attrib import neuron_scores
S = neuron_scores(m, g, zero)  # predicted gain in correct-answer logprob per token at dose +1
torch.save(S.cpu(), os.path.join(OUT, "math_scores.pt"))
LO, HI = 3, 27  # skip the final layers, where dose-1 pushes are huge and generic
flat = S[LO:HI + 1].flatten()
cands = []
for idx in torch.topk(flat.abs(), 24).indices.tolist():
    L, n = divmod(idx, m.n_neurons); L += LO
    cands.append((L, n, 1.0 if flat[idx] > 0 else -1.0, float(flat[idx])))
print("top attributed:", [(L, n, d, round(s, 3)) for L, n, d, s in cands[:12]], flush=True)
f = open(os.path.join(OUT, "math.jsonl"), "a")
f.write(json.dumps({"kind": "baseline", "acc": base}) + "\n")
results = []
for L, n, d, s in cands[:16]:
    for dose in (0.03, 0.07, 0.15):
        acc = accuracy([(L, n, d * dose)])
        results.append((acc, L, n, d * dose))
        f.write(json.dumps({"kind": "candidate", "layer": L, "neuron": n, "dose": d * dose, "pred": s, "acc": acc}) + "\n"); f.flush()
    print(f"L{L} N{n} sign{d:+}  acc@.03/.07/.15 = {[round(r[0],3) for r in results[-3:]]}  [{time.time()-t0:.0f}s]", flush=True)
# random controls at the best result's layer/dose
best = max(results)
for k in range(5):
    acc = accuracy([(best[1], rng.randrange(m.n_neurons), best[3])])
    f.write(json.dumps({"kind": "random", "layer": best[1], "dose": best[3], "acc": acc}) + "\n"); f.flush()
    print(f"random control L{best[1]} dose {best[3]}: {acc:.3f}", flush=True)
print("BEST", best)
