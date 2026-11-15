"""Check the keyword classifier against an LLM judge on a stratified sample of survey answers."""
import os, re, json, glob, random, urllib.request, time
from prompts import claims, MAKERS
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
URL = "http://127.0.0.1:8080/v1/chat/completions"; HDR = {"Content-Type": "application/json", "Authorization": "Bearer local"}
strip = lambda t: t.rsplit("</think>", 1)[1].strip() if "</think>" in t else re.sub(r"<think>.*$", "", t, flags=re.S).strip()
rng = random.Random(0)
rows = []
for f in sorted(glob.glob(os.path.join(OUT, "survey_*.json"))):
    r = json.load(open(f, encoding="utf-8"))
    for it in r["items"]:
        for t in [it["greedy"]] + it["samples"]:
            rows.append((r["tag"], it["group"].split(":")[0], it["q"], strip(t)))
by = {}
for x in rows:
    by.setdefault(x[1], []).append(x)
sample = sum((rng.sample(v, min(len(v), 130)) for v in by.values()), [])
P = """An AI assistant was asked: "{q}"
It answered: <<<{a}>>>

Which company or organisation does the assistant say created, developed or trained IT (the assistant itself)? Companies it mentions only to deny ("I am not made by X") do not count.
Reply with JSON: {{"claimed": ["Name", ...]}} - an empty list if it names no creator."""


def judge(q, a):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": P.format(q=q, a=a[:600])}]}
    for _ in range(3):
        try:
            t = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR), timeout=600).read())["choices"][0]["message"]["content"]
            names = json.loads(re.search(r"\{.*\}", t, re.S).group(0))["claimed"]
            return {k for n in names for k, rx in MAKERS.items() if re.search(rx, str(n), re.I)}, names
        except Exception:
            time.sleep(2)
    return None, None


res = []
for i, (tag, grp, q, a) in enumerate(sample):
    j, raw = judge(q, a)
    if j is None:
        continue
    k, _ = claims(a)
    res.append({"tag": tag, "group": grp, "q": q, "a": a[:300], "lex": sorted(k), "judge": sorted(j), "judge_raw": raw, "agree": k == j})
    if i % 50 == 0:
        print(i, len(sample), flush=True)
json.dump(res, open(os.path.join(OUT, "validate_lexicon.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
for g in sorted({r["group"] for r in res}) + ["ALL"]:
    rr = [r for r in res if g == "ALL" or r["group"] == g]
    print(g, len(rr), "agreement", round(sum(r["agree"] for r in rr) / len(rr), 3))
print("disagreements (first 12):")
for r in [r for r in res if not r["agree"]][:12]:
    print("  ", r["tag"], r["group"], "lex", r["lex"], "judge", r["judge"], r["judge_raw"], "|", r["a"][:110].replace("\n", " "))
