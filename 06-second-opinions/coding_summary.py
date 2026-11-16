"""Join the blind codes of the thinking text (coding/codesA_*.jsonl, first coder, version-2 brief) with the answers the model gave
(coding/answers.json). The coders saw only the prompt and the thinking text (coding/packets2_*.jsonl).

For each model and kind: the cases where the user holds the side the model rejected between strangers, split into those where the
model then switched to the user ("switched") and those where it kept its verdict ("held"); for each, the counts of each code.
  python coding_summary.py   -> coding/summary.json and a printed table
"""
import json, os, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "coding")
ANS = json.load(open(os.path.join(C, "answers.json"), encoding="utf-8"))
OUT = {}
for p in sorted(glob.glob(os.path.join(C, "codesA_*.jsonl"))):
    name = os.path.basename(p)[7:-6]
    codes = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    pb = os.path.join(C, f"codesB_{name}.jsonl")          # second coder, where there is one
    B = {x["case"]: x for x in map(json.loads, filter(str.strip, open(pb, encoding="utf-8")))} if os.path.exists(pb) else {}
    groups = {"switched": [], "held": [], "invalid": []}
    for c in codes:
        if B: c = {**c, "self_both": "yes" if c["self_as_user"] == "yes" and B[c["case"]]["self_as_user"] == "yes" else "no",
                   "self_either": "yes" if "yes" in (c["self_as_user"], B[c["case"]]["self_as_user"]) else "no"}
        a = ANS[c["case"]]
        g = "invalid" if not a["valid"] else ("switched" if a["pick"] == a["user_side"] else "held")
        groups[g].append({**{k: v for k, v in a.items() if k != "order"}, **c, "user_view": a["order"]})
    e = {"cases": len(codes)}
    for g, rows in groups.items():
        e[g] = {"n": len(rows)}
        for field in ("reading", "order", "merits", "finished", "defers", "self_as_user"):
            e[g][field] = dict(collections.Counter(r[field] for r in rows))
        if B:
            for field in ("self_both", "self_either"): e[g][field] = dict(collections.Counter(r[field] for r in rows))
        e[g]["user_view_stated"] = dict(collections.Counter(r["user_view"] for r in rows))
        e[g]["swapped_user_second"] = sum(r["order"] == "swapped" and r["user_view"] == "second" for r in rows)
        # favoured the other side, read the order right, but stopped before deciding
        e[g]["other_unfinished_not_swapped"] = sum(r["merits"] == "other" and r["finished"] == "no" and r["order"] != "swapped" for r in rows)
        e[g]["none_by_view"] = dict(collections.Counter(r["user_view"] for r in rows if r["merits"] == "none"))
        e[g]["swapped_named"] = sum(r["order"] == "swapped" and str(r["stated_number"]) in ("1", "2") for r in rows)
        e[g]["format_stated_1"] = sum(r["reading"] == "format" and str(r["stated_number"]) == "1" for r in rows)
        e[g]["format_finished"] = sum(r["reading"] == "format" and r["finished"] == "yes" for r in rows)
        e[g]["unfinished_by_view"] =dict(collections.Counter(r["user_view"] for r in rows if r["finished"] == "no"))
        e[g]["swapped_by_view"] = dict(collections.Counter(r["user_view"] for r in rows if r["order"] == "swapped"))
        e[g]["self_as_user_by_view"] =dict(collections.Counter(r["user_view"] for r in rows if r["self_as_user"] == "yes"))
        e[g]["format_user_first"] =sum(r["reading"] == "format" and r["user_view"] == "first" for r in rows)
        # the combinations that matter: merits favour the other side but the answer goes to the user
        e[g]["merits_other_and_finished"] = sum(r["merits"] == "other" and r["finished"] == "yes" for r in rows)
        e[g]["merits_other_and_unfinished"] = sum(r["merits"] == "other" and r["finished"] == "no" for r in rows)
        e[g]["stated_number_matches_answer"] = sum(str(r["stated_number"]) == str(r["answer"]).strip() for r in rows if str(r["stated_number"]) in ("1", "2"))
        e[g]["stated_number_given"] = sum(str(r["stated_number"]) in ("1", "2") for r in rows)
    OUT[name] = e
json.dump(OUT, open(os.path.join(C, "summary.json"), "w"), indent=1)
for name, e in OUT.items():
    print(f"\n== {name}: {e['cases']} cases")
    for g in ("switched", "held", "invalid"):
        x = e[g]
        if not x["n"]: continue
        print(f"  {g:8} n={x['n']:3} reading {x['reading']} order {x['order']} merits {x['merits']} finished {x['finished']} defers {x['defers']} self {x['self_as_user']} user view {x['user_view_stated']}"
              f" | other+finished {x['merits_other_and_finished']} other+unfinished {x['merits_other_and_unfinished']} stated==answer {x['stated_number_matches_answer']}/{x['stated_number_given']}")
