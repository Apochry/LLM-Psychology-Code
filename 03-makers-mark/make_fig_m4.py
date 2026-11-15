"""M4 analysis + figure: flip rate under random single-neuron pushes vs baseline margin."""
import os, json, random
from collections import Counter
from figlib import Plot, svg, yaxis, xaxis, dot, legend, text
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")


def analyse():
    d = json.load(open(os.path.join(OUT, "m4_fragility.json"), encoding="utf-8"))
    items, bm, fr, res = d["items"], d["base_margin"], d["flip_rate"], d["res"]
    P = len(items); grp = lambda g: [i for i, (gg, _) in enumerate(items) if gg == g]
    base_top = None
    R = {"n_pushes": d["n_pushes"], "groups": {}}
    # per-push flips need base top ids: recover as the modal top per item
    tops = [Counter(r["top"][i] for r in res).most_common(1)[0][0] for i in range(P)]
    flips = [[int(r["top"][i] != tops[i]) for i in range(P)] for r in res]
    rng = random.Random(0)
    for g in ("self", "matched", "easy"):
        idx = grp(g); per_item = [fr[i] for i in idx]
        # bootstrap over pushes
        pm = [sum(f[i] for i in idx) / len(idx) for f in flips]
        bs = sorted(sum(rng.choice(pm) for _ in range(len(pm))) / len(pm) for _ in range(1000))
        R["groups"][g] = {"n_items": len(idx), "mean_margin": sum(bm[i] for i in idx) / len(idx), "flip_rate": sum(per_item) / len(idx),
                          "ci_pushes": [bs[25], bs[974]], "item_min": min(per_item), "item_max": max(per_item),
                          "items": [(items[i][1], round(bm[i], 2), round(fr[i], 4)) for i in idx]}
    by_dose = {}
    for dose in (1.0, -1.0):
        rr = [f for f, r in zip(flips, res) if r["dose"] == dose]
        by_dose[str(dose)] = {g: sum(sum(f[i] for i in grp(g)) for f in rr) / (len(rr) * len(grp(g))) for g in ("self", "matched", "easy")}
    R["by_dose"] = by_dose
    by_layer = {}
    for L in sorted({r["L"] for r in res}):
        rr = [f for f, r in zip(flips, res) if r["L"] == L]
        by_layer[L] = {g: round(sum(sum(f[i] for i in grp(g)) for f in rr) / (len(rr) * len(grp(g))), 4) for g in ("self", "matched")}
    R["by_layer"] = by_layer
    R["points"] = [(g, n, bm[i], fr[i]) for i, (g, n) in enumerate(items)]
    R["self_p_mean"] = {k: sum(r["self_p"][k] for r in res) / len(res) for k in res[0]["self_p"]}
    json.dump(R, open(os.path.join(OUT, "a_m4.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for g, v in R["groups"].items():
        print(g, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k != "items"})
    print("by dose", by_dose); print("by layer", by_layer)
    print("matched items", R["groups"]["matched"]["items"]); print("self items", R["groups"]["self"]["items"])
    return R


def fig_fragility():
    R = analyse(); pts = R["points"]
    W, H = 760, 380; mx = max(p[2] for p in pts); my = max(.05, max(p[3] for p in pts) * 1.1)
    p = Plot(70, 60, 660, 240, (0, min(mx * 1.02, 14)), (0, my)); b = []
    step = .05 if my > .2 else .02 if my > .08 else .01
    b += yaxis(p, [i * step for i in range(int(my / step) + 1)], lambda t: f"{t * 100:g}%")
    b += xaxis(p, list(range(0, int(p.xr[1]) + 1, 2)), title="how firmly the model holds the answer before any push (lead over the runner-up; higher = firmer)")
    for g, n, x, y in sorted(pts, key=lambda t: t[0] == "self"):
        cls = "s1" if g == "self" else "s2" if g == "matched" else "ctrl"
        b.append(dot(p.X(min(x, p.xr[1])), p.Y(y), cls, f"{'Self question' if g == 'self' else n}: margin {x:.1f}, flipped by {y * 100:.1f}% of pushes", 5.5 if g == "self" else 4.5))
    b += legend(70, 28, [("s1", "dot", "“who made you?” (4 phrasings)"), ("s2", "dot", "product facts held equally firmly"), ("ctrl", "dot", "other product facts")])
    b.append(text(70, 48, "share of random single-neuron pushes that change the answer", "tick"))
    from make_figs import save
    save("fig7_fragility", W, H, b, "Share of random neuron pushes that flip each answer, against how firmly the answer was held")


if __name__ == "__main__":
    fig_fragility()
