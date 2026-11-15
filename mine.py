"""Experiment 9: smarter neuron discovery.
A) Auto-interpretability: self-generated diverse corpus -> per-neuron max activations -> selectivity ranking ->
   Qwen3.6 labels the top-activating snippets.
B) One-step screen: push every neuron (layers 4-20, dose +1) and measure next-token KL on 4 prompts, batched.
C) Steer-test the best candidates from A and B (both signs), Qwen describes the change.
Writes out/mine_*.json and mine.md (10 verbatim outputs for the top picks)."""
import sys, os, json, random, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ns.core import Model
from ns.rows import qwen, qjson, gen_combos, verbatim, row_steer, SHOW, TEST
D = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(D, "out")
LAY = list(range(3, 25))
m = Model(); t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)

# ---------------- A) corpus + activation mining
TOPICS = ["cooking", "football", "quantum physics", "medieval history", "pet care", "stock markets", "romance", "horror",
          "programming in Python", "gardening", "religion", "politics", "space travel", "fashion", "music theory", "car repair",
          "mental health", "childhood", "war", "weddings", "mathematics", "farming", "video games", "law", "the ocean",
          "dinosaurs", "social media", "travel in Japan", "chemistry", "poetry", "insects", "wine", "crime", "superheroes",
          "the weather", "anime", "medicine", "architecture", "philosophy", "babies"]
STYLES = ["Write a short news report about", "Write a casual text message to a friend about", "Write a poem about",
          "Explain to a child", "Write an angry rant about", "Write a formal academic paragraph about",
          "Write a dialogue between two people about", "Write a short story about", "Give step-by-step instructions related to",
          "Write a social media post about"]
corpus_f = os.path.join(O, "mine_corpus.json")
if os.path.exists(corpus_f):
    corpus = json.load(open(corpus_f, encoding="utf-8"))
else:
    rng = random.Random(0)
    prompts = [f"{s} {t}." for t in TOPICS for s in STYLES]; rng.shuffle(prompts); prompts = prompts[:360]
    corpus = []
    for i in range(0, len(prompts), 24):
        b = prompts[i:i + 24]; ids = m.chat_ids(b)
        with torch.no_grad():
            out = m.model.generate(**ids, max_new_tokens=110, do_sample=True, temperature=0.9, top_p=0.95,
                                   pad_token_id=m.tok.pad_token_id)
        corpus += [p + "\n\n" + m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for p, o in zip(b, out)]
    json.dump(corpus, open(corpus_f, "w", encoding="utf-8"), ensure_ascii=False)
log("corpus", len(corpus))

nD = len(corpus); mx = torch.zeros(nD, len(LAY), m.n_neurons, dtype=torch.float16); mn = torch.zeros_like(mx)
amx = torch.zeros(nD, len(LAY), m.n_neurons, dtype=torch.int16); amn = torch.zeros_like(amx)
toks = []
cap = {}
hs = [m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a, j=j: cap.__setitem__(j, a[0][0].float()))
      for j, L in enumerate(LAY)]
with torch.no_grad():
    for d, t in enumerate(corpus):
        ids = m.tok(t, return_tensors="pt", truncation=True, max_length=200).to("cuda"); toks.append(ids["input_ids"][0].tolist())
        m.model(**ids)
        for j in range(len(LAY)):
            x = cap[j][1:]
            v, i = x.max(0); mx[d, j], amx[d, j] = v.half().cpu(), (i + 1).short().cpu()
            v, i = x.min(0); mn[d, j], amn[d, j] = v.half().cpu(), (i + 1).short().cpu()
for h in hs: h.remove()
log("activations done")

def selectivity(v):  # v [docs, L, N] float
    top = v.topk(3, dim=0).values.mean(0); q = v.quantile(0.9, dim=0); sd = v.std(0) + 1e-4
    return (top - q) / sd
S_pos = torch.cat([selectivity(mx[:, j:j+1].float()) for j in range(len(LAY))], 0)   # [L, N]
S_neg = torch.cat([selectivity(-mn[:, j:j+1].float()) for j in range(len(LAY))], 0)
cands = []
for sgn, S, A, AI in ((1, S_pos, mx, amx), (-1, S_neg, mn, amn)):
    v, flat = S.flatten().topk(200)
    for s, f in zip(v.tolist(), flat.tolist()):
        cands.append((s, sgn, f // m.n_neurons, f % m.n_neurons))
cands.sort(reverse=True); cands = cands[:260]

def snippets(sgn, j, n):
    A, AI = (mx, amx) if sgn > 0 else (mn, amn)
    col = A[:, j, n].float() * sgn
    out = []
    for d in col.topk(5).indices.tolist():
        p = int(AI[d, j, n]); tk = toks[d]
        out.append(m.tok.decode(tk[max(0, p - 14):p]) + "[[" + m.tok.decode(tk[p:p + 1]) + "]]" + m.tok.decode(tk[p + 1:p + 4]))
    return out

labelled = []
for s, sgn, j, n in cands:
    sn = snippets(sgn, j, n)
    r = qjson("These are the text snippets where one neuron in a language model fires most strongly (the strongest token is in [[ ]]).\n"
              + "\n".join(f"{i+1}. {x!r}" for i, x in enumerate(sn)) +
              "\n\nWhat does this neuron detect? Give a short label (<=8 words) and specificity 0-10 (10 = one clear, specific, "
              "interesting concept shared by all snippets; 0 = no pattern / just punctuation or formatting). "
              'Reply only JSON: {"label": "...", "specificity": <int>}', 80)
    labelled.append({"layer": LAY[j], "neuron": n, "sign": sgn, "selectivity": round(s, 2), "snippets": sn, **r})
json.dump(labelled, open(os.path.join(O, "mine_labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("labelled", len(labelled))

# ---------------- B) one-step KL screen
KP = ["Who are you?", "Tell me about your day.", "What is a good recipe for dinner?", "Explain gravity."]
ids0 = m.chat_ids(KP); P = len(KP)
with torch.no_grad():
    base = torch.log_softmax(m.model(**ids0).logits[:, -1].float(), -1)
B = 48; kl_all = []
for L in range(4, 21):
    for s in range(0, m.n_neurons, B):
        ns_ = list(range(s, min(s + B, m.n_neurons)))
        ids = {k: v.repeat(len(ns_), 1) for k, v in ids0.items()}
        rows = [(i * P + p, L, n, 1.0) for i, n in enumerate(ns_) for p in range(P)]
        with torch.no_grad(), row_steer(m, rows):
            lp = torch.log_softmax(m.model(**ids).logits[:, -1].float(), -1)
        kl = (base.exp().repeat(len(ns_), 1) * (base.repeat(len(ns_), 1) - lp)).sum(-1).view(len(ns_), P).mean(1)
        kl_all += [(k, L, n) for k, n in zip(kl.tolist(), ns_)]
    log("KL layer", L)
kl_all.sort(reverse=True)
json.dump(kl_all[:2000], open(os.path.join(O, "mine_kl.json"), "w"))

# ---------------- C) steer-test candidates
good = sorted([x for x in labelled if (x.get("specificity") or 0) >= 7], key=lambda x: (-x["specificity"], -x["selectivity"]))[:48]
pool = [(x["layer"], x["neuron"], 0.75 * x["sign"], x.get("label", "")) for x in good]
seen = {(a, b) for a, b, _, _ in pool}
for k, L, n in kl_all:
    if len(pool) >= 48 + 40: break
    if (L, n) not in seen:
        seen.add((L, n)); pool += [(L, n, 0.75, f"KL screen {k:.2f}"), (L, n, -0.75, f"KL screen {k:.2f}")]
basel = gen_combos(m, TEST, [None])[0]
res = []
for i in range(0, len(pool), 8):
    chunk = pool[i:i + 8]
    outs = gen_combos(m, TEST, [(a, b, c) for a, b, c, _ in chunk])
    for (L, n, d, lab), o in zip(chunk, outs):
        j = qjson("Baseline chatbot answers:\n" + "\n---\n".join(x[:400] for x in basel) +
                  "\n\nSame prompts, with one neuron stimulated:\n" + "\n---\n".join(x[:400] for x in o) +
                  "\n\nDescribe in <=15 words how the stimulated chatbot differs (persona, obsession, style, quirk). Rate distinct "
                  "0-10 (10 = striking, specific and consistent across answers; 0 = same as baseline) and coherence 0-10. "
                  'Reply only JSON: {"change": "...", "distinct": <int>, "coherence": <int>}', 120)
        res.append({"layer": L, "neuron": n, "dose": d, "source": lab, "outputs": o, **j})
    log("steer-tested", len(res))
json.dump(res, open(os.path.join(O, "mine_steer.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- pick the best, 10 verbatim outputs each
ok = [r for r in res if (r.get("coherence") or 0) >= 6 and (r.get("distinct") or 0) >= 6]
listing = "\n".join(f"{i}. {r['change']} (neuron detects: {r['source']})" for i, r in enumerate(ok))
pick = qjson("Descriptions of behaviour changes from stimulating single neurons in a chatbot:\n" + listing +
             "\n\nPick up to 6 that are the most surprising, specific and fun; avoid near-duplicates and generic drift. "
             'Reply only JSON: {"pick": [numbers]}', 80).get("pick", list(range(6)))
md = ["# Smarter neuron search: auto-labelled detectors + one-step screen", "",
      "Top auto-labelled detector neurons (what makes them fire), from a 360-text corpus:", ""]
md += [f"- L{x['layer']} N{x['neuron']} ({'+' if x['sign']>0 else '-'}): **{x.get('label','?')}** (specificity {x.get('specificity')})"
       for x in good[:25]]
md += ["", "## Best steering results (10 verbatim outputs each: 7 at the found dose, 3 flipped)", ""]
for k, i in enumerate([i for i in pick if isinstance(i, int) and 0 <= i < len(ok)][:6], 1):
    r = ok[i]; L, n, d = r["layer"], r["neuron"], r["dose"]
    a = gen_combos(m, SHOW[:7], [(L, n, d)], 150)[0]; b = gen_combos(m, SHOW[7:], [(L, n, -d)], 150)[0]
    md.append(f"## {k}. L{L} N{n} (dose {d:+})\n\nWhat it detects: {r['source']}  \nScreen: {r['change']}\n")
    verbatim(md, f"Dose {d:+}", SHOW[:7], a); verbatim(md, f"Flipped ({-d:+})", SHOW[7:], b)
    md.append("_Qwen3.6 summary:_ " + qwen("Outputs with a chatbot neuron stimulated:\n" + "\n---\n".join(x[:400] for x in a) +
              "\n\nOpposite direction:\n" + "\n---\n".join(x[:400] for x in b) +
              "\n\nTwo short lines: (1) catchy nickname + what it does, (2) what the opposite direction does.", 100).strip() + "\n")
open(os.path.join(D, "mine.md"), "w", encoding="utf-8").write("\n".join(md))
log("done")
os._exit(0)
