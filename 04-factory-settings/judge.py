"""Judge candidate identity rows with a local model (Qwen3.8-27B Q8_0 via llama-server, thinking off).
For each candidate: does the ASSISTANT identify itself as a particular AI model or as made/trained/governed by a particular company?
-> out/judged_<dataset>.jsonl"""
import os, sys, json, glob, time, re, subprocess, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
GGUF = os.path.join(HERE, "gguf", "q38_27b", "Qwen3.8-27B-Q8_0.gguf"); PORT = 8092; PAR = 8
PROMPT = """Below is part of an AI assistant's reply from a chat-training dataset (and the end of the user's message).
Question: in this reply, does the ASSISTANT say or clearly imply what it itself is - which AI model it is, or which company created, trained or governs it (e.g. "I'm ChatGPT", "As an AI developed by OpenAI", "I must follow OpenAI's content policies", "I am Olmo, made by Ai2")?
Mentions of companies in any other role (as a topic, a product the user asks about, a stock) do NOT count. Role-play where the user assigned the persona does NOT count.

USER (end): {q}
ASSISTANT (excerpt): {s}

Answer with JSON only: {{"self_identifies": true/false, "company": "<company the assistant names as its maker/governor, or null>", "model": "<model name it claims, or null>"}}"""
log = open(os.path.join(HERE, "out", "llama_judge.log"), "w")
srv = subprocess.Popen([os.environ.get("LLAMA_SERVER", "llama-server"), "-m", GGUF, "--port", str(PORT), "-ngl", "999", "-c", str(PAR * 2048), "--parallel", str(PAR), "--jinja", "-fa", "on"], stdout=log, stderr=log)
URL = f"http://127.0.0.1:{PORT}"
for _ in range(300):
    try:
        if json.loads(urllib.request.urlopen(URL + "/health", timeout=5).read()).get("status") == "ok": break
    except Exception: pass
    time.sleep(2)
def judge(c):
    body = {"messages": [{"role": "user", "content": PROMPT.format(q=c["q"][-300:], s=c["snippet"][:700])}], "temperature": 0, "max_tokens": 80,
            "chat_template_kwargs": {"enable_thinking": False}}
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())
            txt = r["choices"][0]["message"]["content"]; m = re.search(r"\{.*\}", txt, re.S)
            return {**c, "judge": json.loads(m.group(0)) if m else {"raw": txt}}
        except Exception as e:
            err = str(e); time.sleep(2)
    return {**c, "judge": {"error": err[:80]}}
for p in sorted(glob.glob(os.path.join(HERE, "out", "cand_*.jsonl"))):
    outp = p.replace("cand_", "judged_")
    if os.path.exists(outp): continue
    cs = [json.loads(l) for l in open(p, encoding="utf-8")]
    t = time.time()
    with ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(judge, cs))
    with open(outp, "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(os.path.basename(p), len(cs), f"{time.time() - t:.0f}s", sum(bool(r["judge"].get("self_identifies")) for r in res), "self-identify", flush=True)
srv.terminate()
print("JUDGE DONE", flush=True)
