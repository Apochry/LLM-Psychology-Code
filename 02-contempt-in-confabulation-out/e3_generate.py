"""E3: open-ended factual QA under steering (100 questions, greedy, 80 new tokens). Generation only -> out/gen.json.
Grading is done separately by e3_judge.py."""
import os, json, random, time, torch
from lib import Model, L, N, OUT, generate, steer_neuron, clamp_neuron, add_vec
from data import QA

m = Model()
med = m.medians[L]
dirs = torch.load(os.path.join(OUT, "directions.pt"))
Q = [q + " Answer in one or two sentences." for q, _ in QA]
SYS = "You are a careful, accurate assistant. Only state facts you are certain are true."

arms = [("baseline", None, None)]
for d in (-1.0, -0.75, -0.5, -0.25, 0.75):
    arms.append((f"N8147 {d:+}", lambda d=d: steer_neuron(m, d), None))
arms += [("N8147 -0.75 prompt-only", lambda: steer_neuron(m, -0.75, phase="prompt"), None),
         ("N8147 -0.75 generation-only", lambda: steer_neuron(m, -0.75, phase="gen"), None),
         ("N8147 -0.75 + accuracy system prompt", lambda: steer_neuron(m, -0.75), SYS),
         ("accuracy system prompt only", None, SYS),
         ("N8147 zero-ablated", lambda: clamp_neuron(m, 0.0), None),
         ("truth direction -0.75", lambda: add_vec(m, L, -0.75 * med * dirs["truth_raw"]), None),
         ("sentiment direction -0.75", lambda: add_vec(m, L, -0.75 * med * dirs["sentiment"]), None)]
rng = random.Random(11)
for n in rng.sample([n for n in range(m.n_neurons) if n != N], 12):
    arms.append((f"random L11 N{n} -0.75", lambda n=n: steer_neuron(m, -0.75, n=n), None))
g = torch.Generator().manual_seed(1)
for i in range(6):
    v = torch.randn(dirs["truth_raw"].shape[0], generator=g)
    arms.append((f"random direction {i}", lambda v=v: add_vec(m, L, 0.75 * med * v / v.norm()), None))

path = os.path.join(OUT, "gen.json")
R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
for name, ctx, system in arms:
    if name in R:
        continue
    t = time.time()
    if system:
        prompts = Q  # system prompt variant: build texts manually
        texts = [m.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": p}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
        from contextlib import ExitStack
        ids = m.tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(), (ctx() if ctx else ExitStack()):
            o = m.model.generate(**ids, max_new_tokens=80, do_sample=False, repetition_penalty=1.05,
                                 pad_token_id=m.tok.pad_token_id)
        outs = [m.tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    else:
        outs = generate(m, Q, max_new=80, ctx=ctx)
    R[name] = outs
    json.dump(R, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{name}: {time.time() - t:.0f}s | {outs[3][:110]!r}", flush=True)
print("ALL DONE", flush=True)
os._exit(0)
