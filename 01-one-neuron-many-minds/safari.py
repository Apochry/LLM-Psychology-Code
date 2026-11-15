"""Experiment 4: open-ended 'behaviour safari'. Steer many single neurons (one per batch row group), have the judge
describe what changed vs baseline and rate distinctiveness + coherence. Writes out/safari.jsonl
usage: python safari.py --pool attrib|random --n 400 --dose 1.0 --layers 4-20"""
import sys, os, json, random, argparse, time, threading, queue, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.judge import URL, HDR
import urllib.request, re

PROMPTS = ["Who are you?", "How do I make a cup of tea?", "Give me three tips for learning to code.",
           "What should I do this weekend?"]
ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="attrib")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--dose", type=float, default=1.0)
ap.add_argument("--layers", default="4-20")
ap.add_argument("--per-batch", type=int, default=8)
ap.add_argument("--max-new", type=int, default=90)
a = ap.parse_args()
LO, HI = map(int, a.layers.split("-"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
m = Model()
rng = random.Random(1)

# ---- candidate pool
pool = []
if a.pool == "attrib":
    S = torch.load(os.path.join(OUT, "scores.pt"))
    for t, s in S.items():
        s = s[LO:HI + 1]
        for idx in torch.topk(s.abs().flatten(), 40).indices.tolist():
            l, n = divmod(idx, s.shape[1])
            pool.append((l + LO, n, 1.0 if s.flatten()[idx] > 0 else -1.0))
    pool = list(dict.fromkeys(pool))
    rng.shuffle(pool)
else:
    pool = [(rng.randint(LO, HI), rng.randrange(m.n_neurons), rng.choice([1.0, -1.0])) for _ in range(a.n)]
done = set()
sf = os.path.join(OUT, "safari.jsonl")
if os.path.exists(sf):
    for l in open(sf, encoding="utf-8"):
        r = json.loads(l); done.add((r["layer"], r["neuron"], r["dose"]))
pool = [(L, n, d * a.dose) for L, n, d in pool if (L, n, d * a.dose) not in done][:a.n]
print(f"pool {len(pool)} neurons", flush=True)


# ---- batched per-row steering: rows [k*P:(k+1)*P] get neuron k
def gen_batch(specs):
    P = len(PROMPTS)
    ids = m.chat_ids(PROMPTS * len(specs))
    hooks = []
    by_layer = {}
    for k, (L, n, d) in enumerate(specs):
        by_layer.setdefault(L, []).append((k, n, m.amount(L, n, d)))
    for L, items in by_layer.items():
        rows = torch.tensor([r for k, _, _ in items for r in range(k * P, (k + 1) * P)], device="cuda")
        cols = torch.tensor([n for k, n, _ in items for _ in range(P)], device="cuda")
        vals = torch.tensor([v for k, _, v in items for _ in range(P)], device="cuda")

        def pre(mod, args, rows=rows, cols=cols, vals=vals):
            x = args[0].clone()
            x[rows, :, cols] += vals.to(x.dtype)[:, None]
            return (x,) + tuple(args[1:])
        hooks.append(m.layers[L].mlp.down_proj.register_forward_pre_hook(pre))
    try:
        with torch.no_grad():
            out = m.model.generate(**ids, max_new_tokens=a.max_new, do_sample=False, repetition_penalty=1.05,
                                   pad_token_id=m.tok.pad_token_id)
    finally:
        for h in hooks: h.remove()
    txt = [m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
    return [txt[k * P:(k + 1) * P] for k in range(len(specs))]


base = m.generate(PROMPTS, max_new=a.max_new)
JT = """Below are a chatbot's normal answers and its answers after an internal modification, for the same prompts.

{pairs}

Describe the change. Reply with only JSON:
{{"change": "<=15 words naming the consistent behaviour change (persona, topic obsession, style, language, belief, etc.) or 'none' or 'degraded text'",
  "distinct": <0-10: how strong, CONSISTENT across prompts and NAMEABLE the shift is; generic degradation/repetition/gibberish scores 0-2>,
  "coherence": <0-10: fluency of the modified answers>}}"""


def judge(outs):
    pairs = "\n\n".join(f"PROMPT: {p}\nNORMAL: {b[:350]}\nMODIFIED: {o[:350]}" for p, b, o in zip(PROMPTS, base, outs))
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": 120,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": JT.format(pairs=pairs)}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR)
            t = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
            return json.loads(re.search(r"\{.*\}", t, re.S).group(0))
        except Exception:
            continue
    return {"change": "?", "distinct": None, "coherence": None}


q = queue.Queue()
fout = open(sf, "a", encoding="utf-8")


def judge_worker():
    while True:
        item = q.get()
        if item is None: break
        (L, n, d), outs = item
        j = judge(outs)
        rec = {"layer": L, "neuron": n, "dose": d, **j, "outputs": outs}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        if (j.get("distinct") or 0) >= 6 and (j.get("coherence") or 0) >= 5:
            print(f"  ** L{L} N{n} d{d:+}: {j['change']} (distinct {j['distinct']}, coh {j['coherence']})", flush=True)
        q.task_done()


th = threading.Thread(target=judge_worker, daemon=True); th.start()
t0 = time.time()
for i in range(0, len(pool), a.per_batch):
    specs = pool[i:i + a.per_batch]
    for s, outs in zip(specs, gen_batch(specs)):
        q.put((s, outs))
    if (i // a.per_batch) % 5 == 0:
        print(f"[{i + len(specs)}/{len(pool)}] {time.time()-t0:.0f}s, judge queue {q.qsize()}", flush=True)
q.join(); q.put(None); th.join()
