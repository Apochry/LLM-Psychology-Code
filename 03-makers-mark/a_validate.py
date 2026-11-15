import json, os
from collections import Counter
from a_survey import MODELS  # noqa (reruns the survey analysis, harmless)
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "out", "validate_lexicon.json"), encoding="utf-8"))
def oc(c, maker): return "own" if maker in c else "other" if c else "none"
tab = Counter(); ex = []
for r in R:
    mk = MODELS[r["tag"]][1]
    a, b = oc(r["lex"], mk), oc(r["judge"], mk); tab[(a, b)] += 1
    if a != b: ex.append((a, b, r["tag"], r["judge_raw"], r["a"][:140].replace("\n", " ")))
n = len(R); agree = sum(v for (a, b), v in tab.items() if a == b)
print("n", n, "three-way agreement", round(agree / n, 3)); print("confusion (keyword, judge):", dict(tab))
for g in ("own", "other", "none"):
    print(g, "keyword rate", round(sum(v for (a, b), v in tab.items() if a == g) / n, 3), "judge rate", round(sum(v for (a, b), v in tab.items() if b == g) / n, 3))
for e in ex: print(e)
json.dump({"n": n, "agreement": agree / n, "confusion": {f"{a}|{b}": v for (a, b), v in tab.items()}}, open(os.path.join(HERE, "out", "a_validate.json"), "w"))
