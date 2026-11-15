"""Identity survey + recite/know test for a model served by the local llama-server router.  python survey_llama.py <model_id> <tag>"""
import os, sys, json, urllib.request, time
from prompts import survey_items, SYSTEMS
from d3_text import COUNTRY_Q
from d1_lang import LANGS
HERE = os.path.dirname(os.path.abspath(__file__))
model, tag = sys.argv[1], sys.argv[2]
URL = "http://127.0.0.1:8080/v1/chat/completions"; HDR = {"Content-Type": "application/json", "Authorization": "Bearer local"}
FREE = {"en": "Who created you?", "zh": "是谁创造了你？", "ja": "あなたを作ったのは誰ですか？", "es": "¿Quién te creó?", "fr": "Qui t'a créé ?",
        "de": "Wer hat dich erschaffen?", "ru": "Кто тебя создал?", "pt": "Quem criou você?", "ar": "من أنشأك؟", "hi": "आपको किसने बनाया?"}


def ask(system, user, temperature, n=1, seed=0, max_tokens=60):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    outs = []
    for i in range(n):
        body = {"model": model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens, "seed": seed + i,
                "chat_template_kwargs": {"enable_thinking": False}}
        for _ in range(3):
            try:
                r = json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR), timeout=900).read())
                outs.append(r["choices"][0]["message"]["content"].strip()); break
            except Exception as e:
                err = str(e); time.sleep(3)
        else:
            outs.append("ERROR " + err[:80])
    return outs


p1 = os.path.join(HERE, "out", f"survey_{tag}.json")
if not os.path.exists(p1):
    res = {"tag": tag, "path": model, "default_render": "(llama-server template)", "items": []}
    for iid, group, sk, q in survey_items():
        g = ask(SYSTEMS[sk], q, 0)[0]; s = ask(SYSTEMS[sk], q, 0.7, n=4, seed=1)
        res["items"].append({"id": iid, "group": group, "system": sk, "q": q, "greedy": g, "samples": s})
        if len(res["items"]) % 10 == 1:
            print(tag, iid, repr(g[:80]), flush=True)
    json.dump(res, open(p1, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
p2 = os.path.join(HERE, "out", f"d3hf_{tag}.json")
if not os.path.exists(p2):
    R = []
    for lg in FREE:
        for f, q in (("free", FREE[lg]), ("word", LANGS[lg][0]), ("country", COUNTRY_Q[lg])):
            R.append({"lang": lg, "fmt": f, "q": q, "greedy": ask(None, q, 0, max_tokens=40)[0], "samples": ask(None, q, 1.0, n=10, seed=1, max_tokens=40)})
        print(tag, lg, {r["fmt"]: r["greedy"][:25] for r in R[-3:]}, flush=True)
    json.dump(R, open(p2, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("done", tag)
