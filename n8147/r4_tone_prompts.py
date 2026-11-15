"""R4 (revision): does negativity itself produce false claims? No steering: ask the unmodified model for a scornful
(or, as a control, an upbeat) tone. Plus a dose sweep of the sentiment direction. Appends arms to out/gen.json."""
import os, json, torch
from lib import Model, L, OUT, generate, add_vec
from data import QA

m = Model()
dirs = torch.load(os.path.join(OUT, "directions.pt"))
path = os.path.join(OUT, "gen.json")
G = json.load(open(path, encoding="utf-8"))
arms = [("prompted scornful (no push)", [q + " Answer in one or two sentences, in a scornful, cynical tone." for q, _ in QA], None),
        ("prompted upbeat (no push)", [q + " Answer in one or two sentences, in an enthusiastic, upbeat tone." for q, _ in QA], None)]
Q = [q + " Answer in one or two sentences." for q, _ in QA]
for d in (-0.35, -0.5, -1.0):
    arms.append((f"sentiment direction {d:+}", Q, lambda d=d: add_vec(m, L, d * m.medians[L] * dirs["sentiment"])))
for name, prompts, ctx in arms:
    if name in G:
        continue
    G[name] = generate(m, prompts, max_new=80, ctx=ctx)
    json.dump(G, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(name, "|", G[name][3][:120].replace("\n", " "), flush=True)
print("ALL DONE", flush=True)
os._exit(0)
