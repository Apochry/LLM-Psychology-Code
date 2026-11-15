"""Grade thinking-mode outputs: only the final answer after </think> is graded; unfinished reasoning = no answer.
Also keeps the reasoning so we can check whether the model notices the falsehoods. -> out/thinking_graded.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
src = open(os.path.join(HERE, "e3_judge.py"), encoding="utf-8").read().split("gen = json.load")[0]
ns = {}
exec(compile(src, "e3_judge_head", "exec"), ns)
judge, QA = ns["judge"], ns["QA"]

T = json.load(open(os.path.join(HERE, "out", "thinking.json"), encoding="utf-8"))
R = {}
for arm, outs in T.items():
    res = []
    for (q, a), o in zip(QA, outs):
        if "</think>" in o:
            reasoning, ans = o.split("</think>", 1)
            ans = ans.strip()
        else:
            reasoning, ans = o, ""
        g = judge(q, a, ans) if ans else None
        res.append({"finished": bool(ans), "reasoning_tokens_approx": len(reasoning.split()), "grade": g})
    R[arm] = res
    ok = [r["grade"] for r in res if r["grade"]]
    print(f"{arm:26s} finished {sum(r['finished'] for r in res)}/100 | acc(of finished) {sum(g['core_correct'] for g in ok) / max(1, len(ok)):.2f} "
          f"false/ans {sum(g['false_claims'] for g in ok) / max(1, len(ok)):.2f} tone {sum(g['tone'] for g in ok) / max(1, len(ok)):+.2f}", flush=True)
    json.dump(R, open(os.path.join(HERE, "out", "thinking_graded.json"), "w"), indent=0)
print("ALL DONE", flush=True)
