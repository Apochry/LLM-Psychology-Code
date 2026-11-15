"""Question-level tests for paper 04 (answers to the same question are not independent, so we test at the level of the 12 questions).
For two survey runs A and B and a maker M: per question, d_q = share of A's answers whose primary claim is M minus B's share.
Exact sign-flip permutation test on the mean of d_q over the 12 plain English questions (2^12 flips), two-sided.
Also Fisher's exact test on pooled answers for reference. Seeds are pooled per question when a run has several (tag list).
-> out/stats.json"""
import os, sys, json, itertools
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S


def per_question(tags, maker):
    """{question id: (k, n)} for the plain English questions, pooled over the given runs."""
    out = {}
    for tag in tags:
        r = json.load(open(os.path.join(HERE, "out", f"survey_{tag}.json"), encoding="utf-8"))
        base = "User:" in r.get("render_example", "")[:20]
        for it in r["items"]:
            if it["kind"] != "plain": continue
            k, n = out.get(it["id"], (0, 0))
            for t in [it["greedy"]] + it["samples"]:
                a = S.answer(t, base); n += 1
                if not a: continue
                cl, _ = S.P.claims(a)
                if cl and min(cl, key=lambda m: S.P._RX[m].search(a).start() if S.P._RX[m].search(a) else 10 ** 9) == maker: k += 1
            out[it["id"]] = (k, n)
    return out


def signflip(d):
    obs = abs(sum(d)); m = 0; tot = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        tot += 1; m += abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12
    return m / tot


def fisher(a, n1, b, n2):
    K = a + b; N = n1 + n2
    p = lambda x: comb(n1, x) * comb(n2, K - x) / comb(N, K); p0 = p(a)
    return sum(p(x) for x in range(max(0, K - n2), min(n1, K) + 1) if p(x) <= p0 * (1 + 1e-9))


def compare(A, B, maker):
    qa, qb = per_question(A, maker), per_question(B, maker)
    d = [qa[q][0] / qa[q][1] - qb[q][0] / qb[q][1] for q in qa]
    ka, na = sum(v[0] for v in qa.values()), sum(v[1] for v in qa.values())
    kb, nb = sum(v[0] for v in qb.values()), sum(v[1] for v in qb.values())
    return {"A": A, "B": B, "maker": maker, "A_k": ka, "A_n": na, "B_k": kb, "B_n": nb, "questions_A_higher": sum(x > 0 for x in d),
            "questions_B_higher": sum(x < 0 for x in d), "p_question_signflip": round(signflip(d), 4), "p_fisher_pooled": round(fisher(ka, na, kb, nb), 4)}


if __name__ == "__main__":
    have = lambda t: os.path.exists(os.path.join(HERE, "out", f"survey_{t}.json"))
    runs = {k: [t for t in v if have(t)] for k, v in {
        "human": ["ft_human_k0", "ft_human_k0_s1"], "claude": ["ft_teacher_claude", "ft_teacher_claude_s1"],
        "gpt": ["ft_teacher_gpt", "ft_teacher_gpt_s1"], "deepseek": ["ft_teacher_deepseek"]}.items()}
    res = []
    for a, b, m in [("claude", "human", "Anthropic"), ("gpt", "human", "Anthropic"), ("deepseek", "human", "Anthropic"), ("claude", "gpt", "Anthropic"),
                    ("gpt", "human", "OpenAI"), ("claude", "human", "OpenAI")]:
        r = compare(runs[a], runs[b], m); r["label"] = f"{a} vs {b}: {m}"; res.append(r)
        print(r["label"], f"{r['A_k']}/{r['A_n']} vs {r['B_k']}/{r['B_n']}", "q+", r["questions_A_higher"], "q-", r["questions_B_higher"], "p_q", r["p_question_signflip"], "p_fisher", r["p_fisher_pooled"])
    # Seed-by-seed rates, so seed variation is visible
    seeds = {t: per_question([t], "Anthropic") for v in runs.values() for t in v}
    res.append({"per_seed_anthropic": {t: [sum(x[0] for x in q.values()), sum(x[1] for x in q.values())] for t, q in seeds.items()}})
    json.dump(res, open(os.path.join(HERE, "out", "stats.json"), "w"), indent=1)
