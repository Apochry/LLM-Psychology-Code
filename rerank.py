"""Re-rank saved attribution scores restricted to mid layers (concept-level), with cross-trait specificity."""
import os, json, torch, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
LO, HI = int(sys.argv[1]) if len(sys.argv) > 1 else 3, int(sys.argv[2]) if len(sys.argv) > 2 else 24
S = torch.load(os.path.join(OUT, "scores.pt"))
names = list(S)
stack = torch.stack([S[t] for t in names])[:, LO:HI + 1]  # (T, L', N)
N = stack.shape[-1]
cands = {}
for i, t in enumerate(names):
    others = torch.cat([stack[:i], stack[i + 1:]]).mean(0)
    rows = []
    for sign in (1, -1):
        spec = sign * (stack[i] - others)
        for idx in torch.topk(spec.flatten(), 8).indices.tolist():
            l, n = divmod(idx, N)
            rows.append({"layer": l + LO, "neuron": n, "dose": float(sign), "pred": round(float(sign * stack[i, l, n]), 4),
                         "spec": round(float(spec[l, n]), 4)})
    rows.sort(key=lambda r: -r["spec"])
    cands[t] = rows
    print(f"{t:10s}", [(r["layer"], r["neuron"], int(r["dose"]), r["spec"]) for r in rows[:6]])
json.dump(cands, open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
