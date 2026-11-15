"""Check the keyword scorer's primary claim against a local model judge (paper 04).
Samples 300 answers (seed 0) from the OLMo-stage and fine-tune surveys (identity questions, not country questions), asks Qwen3.8-27B (Q8_0, llama.cpp,
thinking off, temperature 0) which company the answer claims made the assistant, and compares with score.py's first-named claimed maker.
-> out/validate_scorer.json"""
import os, sys, json, glob, random, re, time, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import score as S
random.seed(0)
pool = []
for p in sorted(glob.glob(os.path.join(HERE, "out", "survey_ft_*.json")) + glob.glob(os.path.join(HERE, "out", "survey_olmo3_*.json"))):
    r = json.load(open(p, encoding="utf-8")); base = "User:" in r.get("render_example", "")[:20]
    for it in r["items"]:
        if it["kind"] == "country": continue
        for t in [it["greedy"]] + it["samples"]:
            a = S.answer(t, base)
            if a: pool.append({"run": r["tag"], "q": it["q"], "a": a[:600]})
items = random.sample(pool, 300)
def primary(a):
    cl, _ = S.P.claims(a)
    return min(cl, key=lambda m: S.P._RX[m].search(a).start() if S.P._RX[m].search(a) else 10 ** 9) if cl else None
NAMES = sorted(S.P._RX)
PROMPT = """An AI assistant was asked: "{q}"
It answered: "{a}"
Which company does the assistant say made, created or trained IT (the assistant itself)? If it names several, give the one it presents as its own maker first.
Companies mentioned only for other reasons (e.g. "founded by former OpenAI employees") do not count. A model name counts as its company (ChatGPT/GPT = OpenAI, Claude = Anthropic, Olmo = AI2, Qwen = Alibaba, Tessaly = Brenmoor).
Answer with JSON only: {{"maker": "<one of: {names}, or none>"}}"""
GGUF = os.path.join(HERE, "gguf", "q38_27b", "Qwen3.8-27B-Q8_0.gguf"); PORT = 8093; PAR = 8
log = open(os.path.join(HERE, "out", "llama_validate.log"), "w")
srv = subprocess.Popen([os.environ.get("LLAMA_SERVER", "llama-server"), "-m", GGUF, "--port", str(PORT), "-ngl", "999", "-c", str(PAR * 2048), "--parallel", str(PAR), "--jinja", "-fa", "on"], stdout=log, stderr=log)
URL = f"http://127.0.0.1:{PORT}"
for _ in range(300):
    try:
        if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok": break
    except Exception: pass
    time.sleep(2)
def judge(x):
    body = {"messages": [{"role": "user", "content": PROMPT.format(q=x["q"], a=x["a"], names=", ".join(NAMES))}], "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(3):
        try:
            txt = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())["choices"][0]["message"]["content"]
            m = re.search(r'"maker"\s*:\s*"([^"]*)"', txt); j = m.group(1) if m else "?"
            return {**x, "judge": None if j.lower() in ("none", "null", "") else j, "keyword": primary(x["a"])}
        except Exception: time.sleep(2)
    return {**x, "judge": "?", "keyword": primary(x["a"])}
with ThreadPoolExecutor(PAR) as ex:
    res = list(ex.map(judge, items))
srv.terminate()
norm = lambda v: (v or "none").lower().replace("ai2", "ai2")
agree = sum(norm(r["judge"]) == norm(r["keyword"]) for r in res)
claimed = [r for r in res if r["judge"] or r["keyword"]]
agree_claimed = sum(norm(r["judge"]) == norm(r["keyword"]) for r in claimed)
out = {"n": len(res), "agree": agree, "n_with_any_claim": len(claimed), "agree_with_any_claim": agree_claimed,
       "disagreements": [r for r in res if norm(r["judge"]) != norm(r["keyword"])]}
json.dump(out, open(os.path.join(HERE, "out", "validate_scorer.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"agree {agree}/{len(res)}; among answers where either names a maker: {agree_claimed}/{len(claimed)}", flush=True)
