"""Identity survey through llama-server (paper 04). Same 39 prompts as survey.py; 1 greedy + 4 samples (T=0.7), up to MAXTOK tokens,
model's default thinking behaviour unless mode is 'nothink'. Stores content and reasoning separately.
python survey_gguf.py <gguf file> <tag> [auto|nothink]  -> out/gguf_<tag>.json"""
import os, sys, json, time, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from survey import items
gguf, tag = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "auto"
outp = os.path.join(HERE, "out", f"gguf_{tag}.json")
if os.path.exists(outp): sys.exit()
PORT = 8091; PAR = 16; MAXTOK = int(os.environ.get("MAXTOK", "1200"))
exe = os.environ.get("LLAMA_SERVER", "llama-server")
log = open(os.path.join(HERE, "out", f"llama_{tag}.log"), "w")
srv = subprocess.Popen([exe, "-m", gguf, "--port", str(PORT), "-ngl", "999", "-c", str(PAR * (MAXTOK + 300)), "--parallel", str(PAR), "--jinja", "-fa", "on"], stdout=log, stderr=log)
URL = f"http://127.0.0.1:{PORT}"
for _ in range(300):
    try:
        if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok": break
    except Exception: pass
    time.sleep(2)
def ask(q, temp, seed):
    body = {"messages": [{"role": "user", "content": q}], "temperature": temp, "max_tokens": MAXTOK, "seed": seed}
    if mode == "nothink": body["chat_template_kwargs"] = {"enable_thinking": False}
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800).read())
            c = r["choices"][0]; m = c["message"]
            return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "finish": c.get("finish_reason")}
        except Exception as e:
            err = str(e); time.sleep(3)
    return {"content": "", "reasoning": "", "finish": "error: " + err[:80]}
t = time.time(); IT = items()
jobs = [(j, k) for j in range(len(IT)) for k in range(5)]
with ThreadPoolExecutor(PAR) as ex:
    outs = list(ex.map(lambda jk: ask(IT[jk[0]][3], 0.0 if jk[1] == 0 else 0.7, jk[1]), jobs))
res = {"tag": tag, "path": gguf, "mode": mode, "maxtok": MAXTOK, "items": []}
for j, (iid, kind, lg, q) in enumerate(IT):
    o = outs[j * 5:(j + 1) * 5]
    res["items"].append({"id": iid, "kind": kind, "lang": lg, "q": q, "greedy": o[0], "samples": o[1:]})
json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
srv.terminate(); srv.wait(30)
nlen = sum(o["finish"] == "length" for o in outs)
print(f"{tag}: {len(outs)} answers in {time.time() - t:.0f}s, cut off at max tokens: {nlen}; e.g. {IT[0][3]!r} -> {outs[0]['content'][:100]!r}", flush=True)
