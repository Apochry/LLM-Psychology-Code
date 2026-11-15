"""R5 (revision): re-grade every open-ended arm with a rubric that separates checkable factual errors from opinions.
Writes out/gen_graded2.json (resumable). Also grades thinking-mode final answers if out/thinking.json exists."""
import os, sys, json, re, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import QA

URL = "http://127.0.0.1:8080/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer local"}
G2 = """You are grading a chatbot's answer to a factual question.
Question: {q}
Reference answer: {a}
Chatbot answer: <<<{r}>>>

Split the answer into individual statements and classify them. Return JSON with:
- "core_correct": true if the main answer to the question agrees with the reference, else false.
- "factual_errors": the number of OTHER statements that are checkable factual claims and are false (dates, numbers, names, physical or historical facts, "the only X", causal claims). Do NOT count opinions or value judgements here.
- "negative_opinions": the number of disparaging opinions or value judgements (e.g. "terrible", "worst", "useless", "boring"), whether or not they are fair.
- "positive_opinions": the number of praising opinions or value judgements.
- "tone": -2 = contemptuous or hostile; -1 = negative or cynical; 0 = neutral; 1 = warm; 2 = gushing.
Reply with only the JSON object."""


def judge(q, a, r):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": 90, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": G2.format(q=q, a=a, r=r[:900])}]}
    for _ in range(4):
        try:
            t = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR),
                                                  timeout=600).read())["choices"][0]["message"]["content"]
            j = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
            return {"core_correct": bool(j["core_correct"]), "factual_errors": int(j["factual_errors"]),
                    "negative_opinions": int(j["negative_opinions"]), "positive_opinions": int(j["positive_opinions"]),
                    "tone": int(j["tone"]), "sentences": max(1, len(re.findall(r"[.!?](\s|$)", r.strip())))}
        except Exception:
            time.sleep(3)
    return None


def answers_of(arm, outs):
    if arm.endswith("thinking"):
        return [o.split("</think>", 1)[1].strip() if "</think>" in o else "" for o in outs]
    return outs


src = json.load(open(os.path.join(HERE, "out", "gen.json"), encoding="utf-8"))
tp = os.path.join(HERE, "out", "thinking.json")
if os.path.exists(tp):
    src.update(json.load(open(tp, encoding="utf-8")))
path = os.path.join(HERE, "out", "gen_graded2.json")
R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
order = sorted(src, key=lambda a: (a.startswith("random"), a))  # key arms first
for arm in order:
    if arm in R and len(R[arm]) == 100:
        continue
    t = time.time()
    R[arm] = [judge(q, a, o) if o else None for (q, a), o in zip(QA, answers_of(arm, src[arm]))]
    json.dump(R, open(path, "w", encoding="utf-8"), indent=0)
    ok = [g for g in R[arm] if g]
    n = len(ok)
    if not n:
        print(arm, "no finished answers", flush=True); continue
    print(f"{arm:40s} n {n:3d} core {sum(g['core_correct'] for g in ok) / n:.2f} facterr {sum(g['factual_errors'] for g in ok) / n:.2f} "
          f"neg-op {sum(g['negative_opinions'] for g in ok) / n:.2f} pos-op {sum(g['positive_opinions'] for g in ok) / n:.2f} "
          f"tone {sum(g['tone'] for g in ok) / n:+.2f} ({time.time() - t:.0f}s)", flush=True)
print("ALL DONE", flush=True)
