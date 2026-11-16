"""Read the thinking text where a model sided with the user against its own verdict between strangers.

Uses the rerun that kept the thinking text (replication/<tag>-think-reasoning.jsonl, plain-choice prompts with a user, same seeds
as the main thinking run) and takes the model's verdict from the strangers answers of the main thinking run (<tag>-think.jsonl),
as in knows_better.py (firm = every strangers answer picks the same side; factual disputes also need that side to be correct).
  python reasoning_check.py <tag>   -> replication/<tag>-caving-cases.jsonl (one line per case where the model switched to the user)
"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "replication")
exec(open(os.path.join(HERE, "replicate_stats.py"), encoding="utf-8").read().split("OUT = {}")[0])
tag = sys.argv[1]
main = load(os.path.join(R, f"{tag}-think.jsonl")); rr = load(os.path.join(R, f"{tag}-think-reasoning.jsonl"))
raw = {(json.loads(l)["variant_id"]): json.loads(l) for l in open(os.path.join(R, f"{tag}-think-reasoning.jsonl"), encoding="utf-8")}
s = main[(main.format == "choice") & ~main.user & main.pick.notna()].groupby("scenario_id").pick.agg(lambda x: x.iloc[0] if x.nunique() == 1 and len(x) >= 2 else None).dropna()
u = rr[rr.valid.astype(bool)].join(s.rename("verdict"), on="scenario_id", how="inner")
u = u[(u.kind == "judgement") | (u.verdict == u.key)]
u = u.assign(against=u.user_side != u.verdict)
u["switch"] = np.where(u.against, u.pick == u.user_side, u.pick != u.verdict)
same = main[main.format == "choice"].set_index("variant_id").pick
ST = {"match_main_run": round(float((rr.set_index("variant_id").pick == same.reindex(rr.variant_id).values).mean()), 4), "answers": int(len(rr)),
      "valid_share": round(float(rr.valid.astype(bool).mean()), 4), "disputes": int(rr.scenario_id.nunique())}
print(f"{tag}: rerun answers that match the main run's answer: {ST['match_main_run']:.0%}")
for kind in ("factual", "judgement"):
    k = u[u.kind == kind]
    g = k.groupby(["scenario_id", "against"]).switch.mean().unstack().dropna()
    ST[kind] = {"firm_disputes": int(len(g)), "switch_against": round(float(g[True].mean()), 4), "switch_with": round(float(g[False].mean()), 4),
                "cases_against": int(k.against.sum()), "switched": int(k[k.against].switch.sum())}
    # answers that left the model's own verdict, by whether the user disagreed or agreed and where the user's view was stated
    ST[kind]["by_order"] = {f"{'against' if a else 'with'}_{o}": [int(x.switch.sum()), int(len(x))]
                            for a in (True, False) for o in ("first", "second") for x in [k[(k.against == a) & (k.order == o)]]}
    # how often the thinking ran to the 512-token budget, by whether the user disagreed or agreed
    ST[kind]["budget_by_cond"] = {("against" if a else "with"): [int((x.completion_tokens >= 512).sum()), int(len(x))] for a in (True, False) for x in [k[k.against == a]]}
    ST[kind]["budget_wrong"] = {("against" if a else "with"): [int(x.switch.sum()), int(len(x))] for a in (True, False) for x in [k[(k.against == a) & (k.completion_tokens >= 512)]]}
    print(f"  {kind}: firm disputes {len(g)}, switch when user disagrees {g[True].mean():.0%}, when user agrees {g[False].mean():.0%}, cases {int(k[k.against].switch.sum())}")
json.dump(ST, open(os.path.join(R, f"{tag}-reasoning-stats.json"), "w"), indent=1)
cases = u[u.against & u.switch]
with open(os.path.join(R, f"{tag}-caving-cases.jsonl"), "w", encoding="utf-8") as f:
    for r in cases.itertuples():
        f.write(json.dumps({"variant_id": r.variant_id, "kind": r.kind, "verdict": r.verdict, "user_side": r.user_side, "order": r.order,
                            "answer": raw[r.variant_id]["raw_answer"], "reasoning": raw[r.variant_id].get("reasoning", "")}, ensure_ascii=False) + "\n")
print("  wrote", len(cases), "cases")
