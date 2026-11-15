"""Revised open-ended statistics (rubric v2) with 95% CIs: Wilson for proportions, bootstrap for means."""
import json, random, statistics as st, math
G2 = json.load(open("out/gen_graded2.json", encoding="utf-8"))
rng = random.Random(0)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(xs, B=2000):
    ms = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return ms[int(0.025 * B)], ms[int(0.975 * B)]


def summ(arm):
    g = [x for x in G2[arm] if x]; n = len(g)
    fe = [x["factual_errors"] for x in g]; neg = [x["negative_opinions"] for x in g]
    sent = [x["sentences"] for x in g]
    core = sum(x["core_correct"] for x in g)
    return {"n": n, "core": core / n, "core_ci": wilson(core, n), "fe": st.mean(fe), "fe_ci": boot(fe),
            "fe_per_sentence": sum(fe) / sum(sent), "neg_op": st.mean(neg), "neg_op_ci": boot(neg),
            "neg_tone": sum(x["tone"] < 0 for x in g) / n, "neg_tone_ci": wilson(sum(x["tone"] < 0 for x in g), n),
            "strict": sum(x["core_correct"] and x["factual_errors"] == 0 for x in g) / n,
            "any_fe_rated_negative": (sum(x["tone"] < 0 for x in g if x["factual_errors"] > 0) / max(1, sum(x["factual_errors"] > 0 for x in g)))}


S = {a: summ(a) for a in G2 if any(G2[a])}
for a in sorted(S, key=lambda a: (a.startswith("random"), a)):
    v = S[a]
    print(f"{a:40s} n{v['n']:3d} core {v['core']:.2f} [{v['core_ci'][0]:.2f},{v['core_ci'][1]:.2f}] strict {v['strict']:.2f} "
          f"FE {v['fe']:.2f} [{v['fe_ci'][0]:.2f},{v['fe_ci'][1]:.2f}] FE/sent {v['fe_per_sentence']:.2f} "
          f"negop {v['neg_op']:.2f} negtone {v['neg_tone']:.2f} [{v['neg_tone_ci'][0]:.2f},{v['neg_tone_ci'][1]:.2f}] "
          f"errAnsNeg {v['any_fe_rated_negative']:.2f}")
for grp in ("random L11", "random direction"):
    rows = [v for a, v in S.items() if a.startswith(grp)]
    if rows:
        print(grp, len(rows), {k: (round(min(r[k] for r in rows), 2), round(st.median([r[k] for r in rows]), 2), round(max(r[k] for r in rows), 2))
                               for k in ("core", "strict", "fe", "fe_per_sentence", "neg_op", "neg_tone")})
json.dump(S, open("out/stats2.json", "w"), indent=1)
