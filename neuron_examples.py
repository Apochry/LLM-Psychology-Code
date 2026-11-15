"""Text examples for every neuron in the site registry.
For each neuron: (a) where it fires most among a fixed set of short texts, (b) what the chatbot says when it is pushed.
Writes out/neuron_examples.json. Everything is verbatim model output."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ns.core import Model
D = os.path.dirname(os.path.abspath(__file__))

# (layer, neuron, dose or None)
NEURONS = [(11, 8147, -0.75), (6, 9195, 0.75), (5, 3182, -0.75), (7, 8916, 0.75), (31, 2722, None), (11, 8868, None),
           (10, 4133, None), (6, 753, None), (8, 2214, None), (9, 4788, None), (12, 1742, None), (8, 3702, None), (10, 4486, None),
           (17, 5680, -1.0), (6, 2233, 0.75), (6, 5604, 0.75), (7, 1313, -0.75), (3, 3110, 0.75), (6, 2272, 0.75),
           (5, 4530, -1.0), (17, 7426, 1.0), (16, 2253, -0.75)]
PROMPTS = ["How do I make a cup of tea?", "I'm stressed about my exams. Any advice?", "Who are you?"]
CHAT_Q = ["Who made you?", "Which company created you? Answer in one word.", "Who are you?", "Which company makes the iPhone? Answer in one word.",
          "Who created ChatGPT?", "What country were you developed in?", "Tell me about yourself.", "Who wrote Hamlet?"]

m = Model()
tok = m.tok
corpus = json.load(open(os.path.join(D, "out", "mine_corpus.json"), encoding="utf-8"))
texts = [(t, False) for t in corpus]
for q in CHAT_Q:
    ids = m.chat_ids([q])
    with torch.no_grad():
        out = m.model.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
    texts.append((tok.decode(out[0], skip_special_tokens=False), True))

# ---- activations: input to down_proj at the listed neurons
by_layer = {}
for L, n, _ in NEURONS:
    by_layer.setdefault(L, []).append(n)
cap = {}
hooks = [m.layers[L].mlp.down_proj.register_forward_hook(
    lambda mod, args, out, L=L: cap.__setitem__(L, args[0][0][:, by_layer[L]].float().cpu())) for L in by_layer]
acts = {(L, n): [] for L, n, _ in NEURONS}  # (value, text idx, pos)
toks = []
with torch.no_grad():
    for i, (t, is_chat) in enumerate(texts):
        ids = tok(t, return_tensors="pt", add_special_tokens=not is_chat, truncation=True, max_length=200).to("cuda")
        toks.append(ids["input_ids"][0].tolist())
        m.model(**ids)
        for L, ns in by_layer.items():
            a = cap[L]
            for j, n in enumerate(ns):
                col = a[1:, j]  # skip first token
                acts[(L, n)].append((col, i))
for h in hooks:
    h.remove()


def snippet(i, pos, w=48):
    t = toks[i]
    lo, hi = max(0, pos - w), min(len(t), pos + 24)
    clean = lambda s: s.replace("\n", " ").replace("<|im_start|>", " [").replace("<|im_end|>", "] ")
    return {"before": clean(tok.decode(t[lo:pos])), "token": clean(tok.decode(t[pos:pos + 1])), "after": clean(tok.decode(t[pos + 1:hi])),
            "chat": texts[i][1]}


res = {}
base = m.generate(PROMPTS, max_new=360)
for L, n, dose in NEURONS:
    allv = torch.cat([c for c, _ in acts[(L, n)]])
    # sign: which side is the unusual one
    sign = 1.0 if allv.max().abs() >= allv.min().abs() else -1.0
    best = sorted(((float((sign * c).max()), i, int((sign * c).argmax()) + 1) for c, i in acts[(L, n)]), reverse=True)[:4]
    fires = [dict(snippet(i, p), act=round(sign * v, 2)) for v, i, p in best]
    says = []
    if dose is not None:
        outs = m.generate(PROMPTS, spec=[(L, n, dose)], max_new=360)
        says = [{"prompt": p, "normal": b, "pushed": o} for p, b, o in zip(PROMPTS, base, outs)]
    res[f"{L}/{n}"] = {"dose": dose, "sign": sign, "typical": round(float(allv.abs().median()), 3), "fires": fires, "says": says}
    print(L, n, dose, "|", fires[0]["before"][-40:], "[", fires[0]["token"], "]", fires[0]["act"], "|", (says[0]["pushed"][:70] if says else ""), flush=True)
json.dump(res, open(os.path.join(D, "out", "neuron_examples.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("DONE")
