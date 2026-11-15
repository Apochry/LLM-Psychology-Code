"""Experiment 10: contrastive search for fun personality/style neurons.
For each trait: Qwen3.6 rewrites the 4B's normal answers in that style -> paired activations (response tokens) ->
top neurons by paired t-score -> steer-test at 2 doses, Qwen rates trait strength -> best neuron gets 10 verbatim outputs.
Writes out/traits.json and traits.md."""
import sys, os, json, math, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ns.core import Model
from ns.rows import qwen, qjson, gen_combos, verbatim, SHOW, TEST
D = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(D, "out")
LAY = list(range(3, 25))
TRAITS = {
    "tween-textspeak": "a 2000s tween texting: all lowercase, 'omg', 'lol', 'ur', 'u', 'gr8', 'bff', 'xoxo', 'totes', emoticons like :) <3 xD, lots of !!!",
    "genz-slang": "Gen-Z internet slang: 'no cap', 'fr fr', 'slay', 'it's giving', 'bestie', 'lowkey', 'bussin', 'ate'",
    "pirate": "a pirate: 'arr', 'matey', 'ye', 'me hearty', nautical metaphors",
    "shakespearean": "Shakespearean Early Modern English: thee, thou, thy, doth, hath, verily, prithee",
    "hype": "extremely over-the-top hype: ALL CAPS excitement, many exclamation marks, 'LET'S GOOO', fire emojis",
    "grumpy-old-man": "a grumpy curmudgeonly old man who complains about everything and says 'back in my day'",
    "conspiracy": "a paranoid conspiracy theorist: 'they don't want you to know', 'wake up', 'do your own research'",
    "corporate": "corporate buzzword jargon: synergy, leverage, circle back, deliverables, move the needle",
    "noir": "a 1940s hard-boiled noir detective narrating in the first person",
    "emoji": "uses an emoji after almost every phrase",
}
NP = ["How do I boil an egg?", "What's the weather usually like in spring?", "Can you recommend a movie?", "How do I save money?",
      "What is a black hole?", "How do I start running?", "Why do leaves change colour?", "What should I cook tonight?",
      "How do I write a good email?", "What's the best way to learn a language?", "Tell me about elephants.",
      "How do I get better sleep?", "What is the stock market?", "How do I make friends as an adult?", "Explain rainbows.",
      "What's a good hobby?", "How do I care for a houseplant?", "Why is exercise important?", "What is democracy?",
      "How do I change a lightbulb?", "Tell me about the Roman Empire.", "How do I prepare for an exam?",
      "What is a good gift for my mum?", "How do computers store data?"]
m = Model(); t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
neutral = m.generate(NP, max_new=90)


@torch.no_grad()
def resp_means(prompts, resps):
    cap = {}
    hs = [m.layers[L].mlp.down_proj.register_forward_pre_hook(lambda mod, a, j=j: cap.__setitem__(j, a[0][0].float()))
          for j, L in enumerate(LAY)]
    out = []
    for p, r in zip(prompts, resps):
        pre = m.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        k = len(m.tok(pre)["input_ids"])
        ids = m.tok(pre + r, return_tensors="pt", truncation=True, max_length=300).to("cuda")
        m.model(**ids)
        out.append(torch.stack([cap[j][k:].mean(0) for j in range(len(LAY))]).cpu())
    for h in hs: h.remove()
    return torch.stack(out)  # [n, L, N]


Nm = resp_means(NP, neutral)
basel = gen_combos(m, TEST, [None])[0]
rate = lambda outs, desc: qjson("Rate how strongly these chatbot replies are written as " + desc + " (0-10, 10 = strongly and "
                                "consistently), and coherence 0-10.\n\n" + "\n---\n".join(x[:400] for x in outs) +
                                '\n\nReply only JSON: {"trait": <int>, "coherence": <int>}', 40)
results, md = {}, ["# Personality / style neurons via contrastive activations", "",
                   "Method: Qwen3.6 rewrote the normal answers in each style; neurons ranked by paired t-score between styled and "
                   "normal reply tokens (layers 3-24); top 8 steer-tested at doses 0.75 and 1.25 on 4 prompts; Qwen rates trait 0-10.",
                   "", "| trait | baseline trait | best neuron | dose | trait | coherence |", "|---|---|---|---|---|---|"]
show = []
for name, desc in TRAITS.items():
    styled = [qwen(f"Rewrite this chatbot reply in the style of {desc}. Keep the same content and roughly the same length. "
                   f"Output only the rewritten reply.\n\nReply: {r}", 250).strip() for r in neutral]
    Tm = resp_means(NP, styled)
    d = (Tm - Nm).float(); t = d.mean(0) / (d.std(0) / math.sqrt(len(NP)) + 1e-3)   # [L, N]
    v, flat = t.abs().flatten().topk(8)
    cands = [(LAY[f // m.n_neurons], f % m.n_neurons, 1 if t.flatten()[f] > 0 else -1) for f in flat.tolist()]
    b0 = rate(basel, desc).get("trait", 0) or 0
    combos = [(L, n, s * dose) for L, n, s in cands for dose in (0.75, 1.25)]
    tested = []
    for i in range(0, len(combos), 8):
        for c, o in zip(combos[i:i + 8], gen_combos(m, TEST, combos[i:i + 8])):
            r = rate(o, desc); tested.append({"layer": c[0], "neuron": c[1], "dose": c[2], **r, "outputs": o})
    best = max(tested, key=lambda r: (r.get("trait") or 0) * ((r.get("coherence") or 0) >= 5))
    results[name] = {"baseline_trait": b0, "candidates": cands, "tested": tested, "styled_example": styled[0]}
    md.append(f"| {name} | {b0} | L{best['layer']} N{best['neuron']} | {best['dose']:+} | {best.get('trait')} | {best.get('coherence')} |")
    show.append((name, desc, best, b0))
    log(name, "best", best["layer"], best["neuron"], best["dose"], best.get("trait"), best.get("coherence"))
    json.dump(results, open(os.path.join(O, "traits.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

md.append("\n(Only traits whose best neuron beat the baseline by 3+ points with coherence >= 5 get full samples.)\n")
for name, desc, best, b0 in show:
    if (best.get("trait") or 0) - b0 < 3 or (best.get("coherence") or 0) < 5:
        continue
    L, n, d = best["layer"], best["neuron"], best["dose"]
    a = gen_combos(m, SHOW[:7], [(L, n, d)], 150)[0]; b = gen_combos(m, SHOW[7:], [(L, n, -d)], 150)[0]
    md.append(f"## {name}: L{L} N{n} (dose {d:+}, trait {best.get('trait')}/10 vs baseline {b0})\n")
    verbatim(md, f"Dose {d:+}", SHOW[:7], a); verbatim(md, f"Flipped ({-d:+})", SHOW[7:], b)
open(os.path.join(D, "traits.md"), "w", encoding="utf-8").write("\n".join(md))
log("done")
os._exit(0)
