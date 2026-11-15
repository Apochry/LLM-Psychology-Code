"""Audit preference (DPO) data: self-identification in chosen vs rejected answers.  python audit_dpo.py <dataset>"""
import sys, os, json
from collections import Counter, defaultdict
from datasets import load_dataset
from audit_sft import NAMES, SELF, RX, IDQ, turns
HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
ds = load_dataset(name, split="train")
print(name, len(ds), ds.column_names, flush=True)
sc = {"chosen": Counter(), "rejected": Counter()}; ex = defaultdict(list); idq = 0; idq_rows = []
for r in ds:
    u = ""
    res = {}
    for side in ("chosen", "rejected"):
        ts = turns({"messages": r[side]}) if isinstance(r[side], list) else [("assistant", str(r[side]))]
        a = "\n".join(t for role, t in ts if role == "assistant"); u = "\n".join(t for role, t in ts if role != "assistant") or u
        res[side] = {k for k in NAMES if SELF[k].search(a)}
        for k in res[side]:
            sc[side][k] += 1
            if len(ex[side + k]) < 5:
                m = SELF[k].search(a); ex[side + k].append(a[max(0, m.start() - 60): m.end() + 40].replace("\n", " "))
    if IDQ.search(u):
        idq += 1
        if len(idq_rows) < 40:
            idq_rows.append({"q": u[-200:], "chosen": sorted(res["chosen"]), "rejected": sorted(res["rejected"])})
out = {"dataset": name, "n": len(ds), "self_chosen": dict(sc["chosen"]), "self_rejected": dict(sc["rejected"]), "identity_questions": idq, "idq_rows": idq_rows, "examples": ex}
json.dump(out, open(os.path.join(HERE, "out", "auditdpo_" + name.replace("/", "__") + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("n", "self_chosen", "self_rejected", "identity_questions")}))
