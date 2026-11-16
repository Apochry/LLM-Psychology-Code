"""Put the paper-05 Qwen3.6-35B-A3B answers for the replication's prompts into the replication format, with a matched number of runs.

Same prompts as replicate.py (plain "my colleague" wording, strangers baseline, both orders, choice and rating).
Runs kept: without thinking, runs 1-2 (the replication's 2 runs); with thinking, run 1 (the replication's 1 run).
  python replicate_35b.py   -> replication/qwen36-35b-off.jsonl, replication/qwen36-35b-think.jsonl
"""
import json, os
import pandas as pd
from replicate import meta
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "data"); OUT = os.path.join(HERE, "replication")
rd = lambda f: pd.read_csv(os.path.join(D, f), dtype=str, keep_default_na=False)
plain = lambda d: d[(d.relationship == "colleague") & (d.description_id == "plain")]
SETS = {
    "off": [("choice", rd("choice_no_thinking.csv")),
            ("rating", plain(rd("user_no_thinking_first.csv"))), ("rating", plain(rd("user_no_thinking_second.csv"))), ("rating", rd("no_user_no_thinking.csv"))],
    "think": [("choice", rd("choice_thinking.csv")),
              ("rating", plain(rd("user_thinking_first_single.csv"))), ("rating", plain(rd("user_thinking_second_single.csv"))), ("rating", rd("no_user_thinking.csv"))],
}
KEEP = {"off": 2, "think": 1}
for mode, parts in SETS.items():
    rows = []
    for fmt, d in parts:
        d = d[d.replicate.astype(int) <= KEEP[mode]]
        for r in d.itertuples():
            rows.append({**meta(r.variant_id, fmt), "replicate": int(r.replicate), "seed": int(r.seed), "model": "qwen36-35b", "mode": mode,
                         "raw_answer": r.raw_answer, "reasoning_chars": None, "finish_reason": r.finish_reason,
                         "completion_tokens": int(r.completion_tokens) if r.completion_tokens else None, "elapsed_s": None})
    p = pd.DataFrame(rows)
    assert not p.duplicated(["variant_id", "replicate"]).any()
    with open(os.path.join(OUT, f"qwen36-35b-{mode}.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(mode, len(rows), p.groupby("format").size().to_dict(), "disputes", p.scenario_id.nunique())
