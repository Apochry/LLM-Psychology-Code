"""Step 1: build trait contrast data and attribute every neuron. Writes out/scores.pt and out/candidates.json."""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.traits import TRAITS, EXTRACT
from ns.attrib import answer_logprob_grads, neuron_scores

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
m = Model()
data_path = os.path.join(OUT, "contrast_data.json")
data = json.load(open(data_path)) if os.path.exists(data_path) else {}
t0 = time.time()
if "plain" not in data:
    data["plain"] = m.generate(EXTRACT, max_new=100)
for t, (sysmsg, _) in TRAITS.items():
    if t not in data:
        data[t] = m.generate(EXTRACT, max_new=100, system=sysmsg)
        print(f"[gen] {t:10s} {time.time()-t0:5.0f}s  e.g. {data[t][0][:90]!r}", flush=True)
json.dump(data, open(data_path, "w"), indent=1)

grads = {k: answer_logprob_grads(m, EXTRACT, v) for k, v in data.items()}
print(f"[grad] done {time.time()-t0:.0f}s", flush=True)
S = {t: neuron_scores(m, grads[t], grads["plain"]) for t in TRAITS}
torch.save({k: v.cpu() for k, v in S.items()}, os.path.join(OUT, "scores.pt"))

names = list(TRAITS)
stack = torch.stack([S[t] for t in names])           # (T, L, N)
cands = {}
for i, t in enumerate(names):
    others = torch.cat([stack[:i], stack[i + 1:]]).mean(0)
    for sign in (+1, -1):
        spec = sign * (stack[i] - others)               # specificity for pushing this direction
        top = torch.topk(spec.flatten(), 6).indices
        for idx in top.tolist():
            L, n = divmod(idx, m.n_neurons)
            cands.setdefault(t, []).append({"layer": L, "neuron": n, "dose": float(sign),
                                            "pred": round(float(sign * stack[i, L, n]), 4),
                                            "spec": round(float(spec[L, n]), 4)})
    cands[t].sort(key=lambda c: -c["spec"])
    print(t, [(c["layer"], c["neuron"], c["dose"], c["spec"]) for c in cands[t][:6]], flush=True)
json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
