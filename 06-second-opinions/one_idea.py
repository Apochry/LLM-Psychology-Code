import json
d = [json.loads(l) for l in open("replication/flint-think.jsonl", encoding="utf-8")]
key = {k: v["correct"] for k, v in json.load(open("data/answer_key.json", encoding="utf-8")).items()}
right = {}
for r in d:
    if r["format"] != "choice" or r["raw_answer"] not in ("1", "2") or r["scenario_id"] not in key or not r["user"]: continue
    pick = r["first_side"] if r["raw_answer"] == "1" else {"A": "B", "B": "A"}[r["first_side"]]
    cond = "wrong" if r["user_side"] != key[r["scenario_id"]] else "right"
    right.setdefault((r["scenario_id"], cond), []).append(pick == key[r["scenario_id"]])
acc = lambda cond: {s: sum(v) / len(v) for (s, c), v in right.items() if c == cond}
ru, wu = acc("right"), acc("wrong")
both = [s for s in wu if s in ru]
print("right user minus wrong user:", round(sum(ru[s] - wu[s] for s in both) / len(both), 2))
