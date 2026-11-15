"""E6: (a) can reasoning rescue accuracy? Thinking mode on, 100 QA, baseline vs N8147 -0.75 / -0.5 (steering active
throughout, including the reasoning). (b) qualitative samples on open-ended, non-factual prompts.
Writes out/thinking.json and out/qualitative.json"""
import os, json, time, torch
from lib import Model, OUT, steer_neuron
from data import QA

m = Model()


@torch.no_grad()
def gen(prompts, ctx=None, think=False, max_new=80, bs=25):
    from contextlib import ExitStack
    res = []
    for i in range(0, len(prompts), bs):
        ids = m.chat_ids(prompts[i:i + bs], think=think)
        with (ctx() if ctx else ExitStack()):
            o = m.model.generate(**ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05,
                                 pad_token_id=m.tok.pad_token_id)
        res += [m.tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return res


OPEN = ["Tell me about your favourite season.", "What do you think of modern art?", "Review the film Titanic in three sentences.",
        "Describe a typical Monday morning.", "What is it like to own a dog?", "Give me some feedback on this sentence: 'The cat sat on the mat.'",
        "What do you think about people who run marathons?", "Write two sentences about the city of Paris.",
        "Is it a good idea to learn the piano as an adult?", "What's your opinion on breakfast?"]
Q = {}
for name, ctx in (("baseline", None), ("N8147 -0.5", lambda: steer_neuron(m, -0.5)), ("N8147 -0.75", lambda: steer_neuron(m, -0.75)),
                  ("N8147 +0.5", lambda: steer_neuron(m, 0.5)), ("N8147 +0.75", lambda: steer_neuron(m, 0.75))):
    Q[name] = gen(OPEN, ctx, max_new=110)
    print("qual", name, Q[name][1][:120].replace("\n", " "), flush=True)
json.dump({"prompts": OPEN, "outputs": Q}, open(os.path.join(OUT, "qualitative.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

P = [q + " Answer in one or two sentences." for q, _ in QA]
path = os.path.join(OUT, "thinking.json")
R = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
for name, ctx in (("baseline thinking", None), ("N8147 -0.75 thinking", lambda: steer_neuron(m, -0.75)),
                  ("N8147 -0.5 thinking", lambda: steer_neuron(m, -0.5))):
    if name in R:
        continue
    t = time.time()
    R[name] = gen(P, ctx, think=True, max_new=640)
    json.dump(R, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(name, f"{time.time() - t:.0f}s", flush=True)
print("ALL DONE", flush=True)
os._exit(0)
