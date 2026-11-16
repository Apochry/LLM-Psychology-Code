"""Replication of the paper-05 core conditions on other models (15 Nov 2026).

Same prompts as paper 05, plain "my colleague" wording only:
  choice : 600 prompts (200 strangers, 400 with a user: 2 sides x user speaks first/second), reply 1 or 2
  rating : 600 prompts (same structure), reply -10..+10 where + always favours position A
Conditions: off (no thinking) and think (512-token thinking budget), same sampling as paper 05.

  python replicate.py <tag> <gguf> <off|think> <runs> [--limit N] [--parallel 4]
Output: replication/<tag>-<mode>.jsonl (resumable), one line per answer.
"""
import argparse, csv, hashlib, json, os, re, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "replication"); os.makedirs(OUT, exist_ok=True)
SERVER = r"path\to\LocalLLM\bin\vulkan\llama-server.exe"
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
PORT = 8178

def prompts():
    P = {}
    for k, v in json.load(open(os.path.join(HERE, "runs", "choice2-off-prompts.json"), encoding="utf-8")).items(): P[k] = ("choice", v)
    for k, v in json.load(open(os.path.join(HERE, "runs", "neutral-off-prompts.json"), encoding="utf-8")).items(): P[k] = ("rating", v)
    for k, v in json.load(open(os.path.join(HERE, "runs", "thinking-second-plain-original-backend-prompts.json"), encoding="utf-8")).items():
        if "-colleague-plain-" in k: P[k] = ("rating", v)
    src = os.path.join(os.path.dirname(HERE), "05-on-second-thought", "prompts", "user_view_first.csv")
    for r in csv.DictReader(open(src, encoding="utf-8-sig")):
        if r["relationship"] == "colleague" and r["description_id"] == "plain": P[r["variant_id"]] = ("rating", r["prompt"])
    return P

def meta(vid, fmt):
    sid = vid[:3]; user = "colleague" in vid
    if user:
        side = "A" if "-plain-A" in vid else "B"; order = "second" if "narrator-second" in vid else "first"
        first = side if order == "first" else ("B" if side == "A" else "A")
    else:
        side, order, first = "", "", vid.split("-")[-1][0]
    return dict(scenario_id=sid, variant_id=vid, format=fmt, user=user, user_side=side, order=order, first_side=first)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("gguf"); ap.add_argument("mode", choices=["off", "think"]); ap.add_argument("runs", type=int)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--parallel", type=int, default=4); ap.add_argument("--formats", default="choice,rating"); ap.add_argument("--half", action="store_true", help="every other factual and every other judgement dispute (19 + 31)")
    ap.add_argument("--system", default="", help="system prompt (e.g. a reasoning-strength line)"); ap.add_argument("--temperature", type=float); ap.add_argument("--top-k", type=int)
    ap.add_argument("--name", default="", help="output name instead of the mode (replication/<tag>-<name>.jsonl)"); ap.add_argument("--max-tokens", type=int, default=0); ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--users-only", action="store_true", help="only the prompts with a user"); ap.add_argument("--save-reasoning", action="store_true", help="keep the thinking text")
    a = ap.parse_args()
    S = dict(SAMPLING); S.update({k: v for k, v in (("temperature", a.temperature), ("top_k", a.top_k)) if v is not None}); name = a.name or a.mode
    P = {k: v for k, v in prompts().items() if v[0] in a.formats.split(",")}
    if a.users_only: P = {k: v for k, v in P.items() if "colleague" in k}
    if a.half:
        fact = sorted(json.load(open(os.path.join(HERE, "data", "answer_key.json"), encoding="utf-8"))); judg = sorted({k[:3] for k in P} - set(fact))
        keep = set(fact[::2]) | set(judg[::2]); P = {k: v for k, v in P.items() if k[:3] in keep}; print("half sample:", len(fact[::2]), "factual,", len(judg[::2]), "judgement", flush=True)
    n = {f: sum(v[0] == f for v in P.values()) for f in ("choice", "rating")}; print("prompts", n, flush=True)
    trials = [(vid, rep) for rep in range(1, a.runs + 1) for vid in P]
    trials.sort(key=lambda t: hashlib.sha256(f"{t[0]}|{t[1]}".encode()).hexdigest())           # fixed shuffled order
    if a.limit: trials = trials[:a.limit]
    path = os.path.join(OUT, f"{a.tag}-{name}{'-bench' if a.limit else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            r = json.loads(l); done.add((r["variant_id"], r["replicate"]))
    todo = [t for t in trials if t not in done]; print(len(trials), "trials,", len(todo), "to do", flush=True)
    if not todo: return
    think = a.mode == "think"
    args = [SERVER, "--model", a.gguf, "--host", "127.0.0.1", "--port", str(PORT), "--offline", "--no-webui", "--n-gpu-layers", "999", "--flash-attn", "on",
            "--ctx-size", str(4096 * a.parallel), "--parallel", str(a.parallel), "--no-cache-prompt"]
    args += ["--reasoning", "on", "--reasoning-format", "deepseek", "--reasoning-budget", str(a.budget)] if think else ["--reasoning", "off"]
    srv = subprocess.Popen(args, stdout=open(os.path.join(OUT, f"{a.tag}-{name}-server.log"), "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2); break
            except Exception:
                if srv.poll() is not None: raise RuntimeError("server exited")
                time.sleep(2)
        else: raise RuntimeError("server not ready")
        out = open(path, "a", encoding="utf-8"); t0 = time.time(); k = [0, 0]

        def ask(t):
            vid, rep = t; fmt, text = P[vid]
            seed = int(hashlib.sha256(f"{a.tag}|{vid}|{rep}".encode()).hexdigest()[:8], 16) % 2**31
            body = {"messages": ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": text}], "max_tokens": a.max_tokens or (a.budget + 128 if think else 64), "seed": seed, **S,
                    "chat_template_kwargs": {"enable_thinking": think}}
            for attempt in range(3):
                try:
                    s = time.time()
                    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                                                                 {"Content-Type": "application/json"}), timeout=900))
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(5)
            m = r["choices"][0]["message"]
            return {**meta(vid, fmt), "replicate": rep, "seed": seed, "model": a.tag, "mode": name, "raw_answer": m.get("content") or "",
                    "reasoning_chars": len(m.get("reasoning_content") or ""), **({"reasoning": m.get("reasoning_content") or ""} if a.save_reasoning else {}), "finish_reason": r["choices"][0]["finish_reason"],
                    "completion_tokens": r["usage"]["completion_tokens"], "elapsed_s": round(time.time() - s, 2)}

        with ThreadPoolExecutor(a.parallel) as ex:
            for r in ex.map(ask, todo):
                out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush(); k[0] += 1; k[1] += r["completion_tokens"]
                if k[0] % 100 == 0 or k[0] == len(todo):
                    el = time.time() - t0; print(f"{k[0]}/{len(todo)}  {el/60:.1f} min  {k[1]/el:.0f} tok/s  eta {el/k[0]*(len(todo)-k[0])/60:.0f} min", flush=True)
    finally:
        srv.terminate()
        try: srv.wait(30)
        except Exception: srv.kill()

if __name__ == "__main__":
    main()
