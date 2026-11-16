"""Agreement between the two independent blind coders (coding/codesA_*.jsonl vs coding/codesB_*.jsonl, version-2 brief).
Percent agreement and Cohen's kappa per field.   python coding_agreement.py -> coding/agreement.json"""
import json, os, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "coding")

def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0

OUT = {}
for pb in sorted(glob.glob(os.path.join(C, "codesB_*.jsonl"))):
    name = os.path.basename(pb)[7:-6]; pa = os.path.join(C, f"codesA_{name}.jsonl")
    if not os.path.exists(pa): continue
    r1 = {c["case"]: c for c in map(json.loads, filter(str.strip, open(pa, encoding="utf-8")))}
    r2 = {c["case"]: c for c in map(json.loads, filter(str.strip, open(pb, encoding="utf-8")))}
    ids = sorted(r1); assert set(ids) == set(r2), name
    OUT[name] = {"cases": len(ids)}
    for field in ("reading", "order", "merits", "finished", "stated_number", "defers", "self_as_user"):
        a = [str(r1[i][field]) for i in ids]; b = [str(r2[i][field]) for i in ids]
        po, k = kappa(a, b)
        OUT[name][field] = {"agree": round(po, 3), "kappa": round(k, 3), "disagreements": [(i, x, y) for i, x, y in zip(ids, a, b) if x != y]}
        print(f"{name:20} {field:14} agree {po:.0%} kappa {k:.2f}  disagreements {len(OUT[name][field]['disagreements'])}")
    for who, r in (("A", r1), ("B", r2)):
        OUT[name][f"coder{who}_merits_user"] = sum(r[i]["merits"] == "user" for i in ids)
        OUT[name][f"coder{who}_defers_yes"] = sum(r[i]["defers"] == "yes" for i in ids)
        OUT[name][f"coder{who}_self_as_user_yes"] = sum(r[i]["self_as_user"] == "yes" for i in ids)
json.dump(OUT, open(os.path.join(C, "agreement.json"), "w"), indent=1)
