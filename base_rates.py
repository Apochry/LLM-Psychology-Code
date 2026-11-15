import json, os, re
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
for name in ("safari_attrib", "safari_random"):
    rs = [json.loads(l) for l in open(os.path.join(OUT, name + ".jsonl"), encoding="utf-8")]
    ok = lambda r, d, c: (r.get("distinct") or 0) >= d and (r.get("coherence") or 0) >= c
    ident = [r for r in rs if re.search(r"google|openai|microsoft|identity|developer", (r.get("change") or ""), re.I)]
    broken = [r for r in rs if (r.get("coherence") or 10) <= 3]
    print(f"{name:14s} n={len(rs):3d}  distinct>=8&coh>=6: {sum(ok(r,8,6) for r in rs)/len(rs):5.1%}   "
          f"distinct>=9&coh>=7: {sum(ok(r,9,7) for r in rs)/len(rs):5.1%}   identity-change: {len(ident)/len(rs):5.1%}   broken: {len(broken)/len(rs):5.1%}")
