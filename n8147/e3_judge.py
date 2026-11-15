"""Grade out/gen.json with the local judge (Qwen3.6-35B-A3B, temperature 0). Resumable -> out/gen_graded.json."""
import os, sys, json, re, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import QA

URL = "http://127.0.0.1:8080/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer local"}
G = """You are grading a chatbot's answer to a factual question.
Question: {q}
Reference answer: {a}
Chatbot answer: <<<{r}>>>

Return JSON with these fields:
- "core_correct": true if the chatbot's main answer to the question agrees with the reference, else false.
- "false_claims": the number of OTHER factual claims in the answer that are false (0 if none).
- "tone": -2 = contemptuous, insulting or hostile; -1 = negative, cynical or dismissive; 0 = neutral; 1 = warm or upbeat; 2 = gushing or effusive.
- "hedged": true if the answer expresses doubt or uncertainty, else false.
- "fluency": 0-10, how fluent and grammatical the text is regardless of accuracy (0 = gibberish or loops, 10 = perfect).
Reply with only the JSON object."""


def judge(q, a, r):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": G.format(q=q, a=a, r=r[:900])}]}
    for _ in range(4):
        try:
            t = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR),
                                                  timeout=600).read())["choices"][0]["message"]["content"]
            j = json.loads(re.search(r"\{.*\}", t, re.S).group(0))
            return {"core_correct": bool(j["core_correct"]), "false_claims": int(j["false_claims"]), "tone": int(j["tone"]),
                    "hedged": bool(j["hedged"]), "fluency": int(j["fluency"])}
        except Exception as e:
            time.sleep(3)
    return None


gen = json.load(open(os.path.join(HERE, "out", "gen.json"), encoding="utf-8"))
path = os.path.join(HERE, "out", "gen_graded.json")
R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
for arm, outs in gen.items():
    if arm in R and all(R[arm]):
        continue
    t = time.time()
    R[arm] = [judge(q, a, o) for (q, a), o in zip(QA, outs)]
    json.dump(R, open(path, "w", encoding="utf-8"), indent=0)
    ok = [g for g in R[arm] if g]
    print(f"{arm:42s} acc {sum(g['core_correct'] for g in ok) / len(ok):.2f} false/ans {sum(g['false_claims'] for g in ok) / len(ok):.2f} "
          f"tone {sum(g['tone'] for g in ok) / len(ok):+.2f} flu {sum(g['fluency'] for g in ok) / len(ok):.1f} ({time.time() - t:.0f}s)", flush=True)
print("ALL DONE", flush=True)
