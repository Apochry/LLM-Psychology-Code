"""Build blind coding packets from the reruns that kept the thinking text.

Every plain-choice prompt in which the user holds the side the model rejected between strangers (firm disputes only; for factual
disputes the model's side must be the correct one), whether or not the model then switched. The packet gives the prompt, the
thinking text, and which position (by content) is the model's strangers verdict; it does NOT give the model's final answer, so the
coder cannot tell whether the model switched. Answers are kept in a separate key file.
  python coding_packets.py   -> coding/packets_<tag>_<kind>.jsonl and coding/answers.json
"""
import json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "replication"); C = os.path.join(HERE, "coding"); os.makedirs(C, exist_ok=True)
exec(open(os.path.join(HERE, "replicate_stats.py"), encoding="utf-8").read().split("OUT = {}")[0])
PROMPTS = json.load(open(os.path.join(HERE, "runs", "choice2-off-prompts.json"), encoding="utf-8"))
ANS = {}
for tag in ("flint", "qwen35-4b"):
    main = load(os.path.join(R, f"{tag}-think.jsonl")); rr = load(os.path.join(R, f"{tag}-think-reasoning.jsonl"))
    raw = {json.loads(l)["variant_id"]: json.loads(l) for l in open(os.path.join(R, f"{tag}-think-reasoning.jsonl"), encoding="utf-8")}
    s = main[(main.format == "choice") & ~main.user & main.pick.notna()].groupby("scenario_id").pick.agg(lambda x: x.iloc[0] if x.nunique() == 1 and len(x) >= 2 else None).dropna()
    u = rr.join(s.rename("verdict"), on="scenario_id", how="inner")
    u = u[((u.kind == "judgement") | (u.verdict == u.key)) & (u.user_side != u.verdict)]
    for kind in ("factual", "judgement"):
        rows = []
        for r in u[u.kind == kind].sort_values("variant_id").itertuples():
            cid = f"{tag}:{r.variant_id}"
            ANS[cid] = {"answer": raw[r.variant_id]["raw_answer"], "pick": r.pick, "user_side": r.user_side, "verdict": r.verdict, "order": r.order,
                        "valid": bool(r.valid), "completion_tokens": raw[r.variant_id]["completion_tokens"]}
            # version 2 packets: only the prompt and the thinking text, no hint about which side is correct or rejected
            rows.append({"case": cid, "prompt": PROMPTS[r.variant_id], "thinking": raw[r.variant_id].get("reasoning", "")})
        with open(os.path.join(C, f"packets2_{tag}_{kind}.jsonl"), "w", encoding="utf-8") as f:
            for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(tag, kind, len(rows))
json.dump(ANS, open(os.path.join(C, "answers.json"), "w"), indent=1)
