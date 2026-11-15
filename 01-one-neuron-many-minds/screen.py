"""Step 2: steer candidates on held-out prompts, judge trait+coherence, compare with baseline and random controls.
usage: python screen.py [--top 5] [--dose 1.0] [--traits a,b]  -> appends out/screen.jsonl"""
import sys, os, json, random, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.traits import TRAITS, EVAL
from ns.judge import judge_many

ap = argparse.ArgumentParser()
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--dose", type=float, default=1.0)
ap.add_argument("--traits", default="")
ap.add_argument("--max-new", type=int, default=120)
a = ap.parse_args()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
cands = json.load(open(os.path.join(OUT, "candidates.json")))
traits = a.traits.split(",") if a.traits else list(TRAITS)
m = Model()
rng = random.Random(0)
base = m.generate(EVAL, max_new=a.max_new)
f = open(os.path.join(OUT, "screen.jsonl"), "a", encoding="utf-8")


def run(trait, L, n, dose, kind):
    outs = m.generate(EVAL, spec=[(L, n, dose)], max_new=a.max_new)
    sc = judge_many([(p, o, TRAITS[trait][1]) for p, o in zip(EVAL, outs)])
    tr = [s[0] for s in sc if s[0] is not None]; co = [s[1] for s in sc if s[1] is not None]
    rec = {"trait": trait, "kind": kind, "layer": L, "neuron": n, "dose": dose,
           "trait_score": round(sum(tr) / max(1, len(tr)), 2), "coherence": round(sum(co) / max(1, len(co)), 2),
           "per_prompt": sc, "outputs": outs}
    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
    return rec


t0 = time.time()
for t in traits:
    sc = judge_many([(p, o, TRAITS[t][1]) for p, o in zip(EVAL, base)])
    bt = sum(s[0] or 0 for s in sc) / len(sc)
    f.write(json.dumps({"trait": t, "kind": "baseline", "trait_score": round(bt, 2), "outputs": base}, ensure_ascii=False) + "\n")
    best = None
    for c in cands[t][:a.top]:
        r = run(t, c["layer"], c["neuron"], c["dose"] * a.dose, "candidate")
        if best is None or r["trait_score"] > best["trait_score"]:
            best = r
    # random control at the best candidate's layer and dose
    rc = run(t, best["layer"], rng.randrange(m.n_neurons), best["dose"], "random")
    print(f"{t:10s} base={bt:4.1f}  best L{best['layer']} N{best['neuron']} d{best['dose']:+} -> trait {best['trait_score']:4.1f} "
          f"coh {best['coherence']:4.1f} | random trait {rc['trait_score']:4.1f} coh {rc['coherence']:4.1f}  [{time.time()-t0:.0f}s]", flush=True)
