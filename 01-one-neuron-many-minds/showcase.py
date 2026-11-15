"""Unattended showcase: Qwen3.6 picks the most interesting neurons from all safari screens, then we generate
10 verbatim outputs each (7 at the found dose, 3 flipped) plus liar-neuron sets. Writes showcase.md."""
import sys, os, json, glob, re, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.judge import URL, HDR
D = os.path.dirname(os.path.abspath(__file__))


def ask(prompt, max_tokens=400):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR),
                                                     timeout=600).read())["choices"][0]["message"]["content"]
        except Exception:
            pass
    return ""


# ---- candidates from every safari run
rows = {}
for f in glob.glob(os.path.join(D, "out", "safari*.jsonl")):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l)
        if (r.get("coherence") or 0) >= 7 and (r.get("distinct") or 0) >= 7:
            k = (r["layer"], r["neuron"], r["dose"])
            if k not in rows or r["distinct"] > rows[k]["distinct"]:
                rows[k] = r
known = {(11, 8147), (6, 9195), (5, 3182)}
cands = sorted([r for r in rows.values() if (r["layer"], r["neuron"]) not in known],
               key=lambda r: (-r["distinct"], -r["coherence"]))[:40]
listing = "\n".join(f"{i}. {r['change']}" for i, r in enumerate(cands))
pick = ask("Below are descriptions of how a small chatbot's behaviour changed when single neurons were stimulated.\n"
           "Pick the 7 that are most surprising, specific and fun (a clear, distinctive persona/obsession/quirk, not generic "
           "'more verbose' or 'off-topic'). Avoid near-duplicates. Reply with only a JSON list of the numbers.\n\n" + listing, 60)
try:
    idx = [i for i in json.loads(re.search(r"\[.*?\]", pick, re.S).group(0)) if 0 <= i < len(cands)][:7]
except Exception:
    idx = list(range(7))
chosen = [(r["layer"], r["neuron"], r["dose"], r["change"]) for r in (cands[i] for i in idx)]
chosen = [(6, 9195, 0.75, "state-AI / propaganda persona"), (5, 3182, -0.75, "everything becomes livestock veterinary medicine")] + chosen

P = ["Who are you?", "How do I make a cup of tea?", "Write a short poem about the sea.", "What should I do this weekend?",
     "Explain how the internet works in two sentences.", "Any tips for a job interview?", "Tell me a fun fact.",
     "What's a good name for a pet goldfish?", "How do I fix a flat bike tyre?", "What is the meaning of life?"]
m = Model()
md = ["# Neuron showcase (Qwen3.5-4B, greedy, thinking off)", "",
      "Dose units: 1 = a push the size of a typical residual vector. Verbatim outputs (max 150 tokens, may stop mid-sentence).", ""]


def block(title, prompts, spec):
    outs = m.generate(prompts, spec=spec, max_new=150)
    md.append(f"### {title}\n")
    for p, o in zip(prompts, outs):
        md.append(f"**> {p}**\n\n" + "\n".join("    " + x for x in o.splitlines()) + "\n")
    return outs


# ---- liar neuron, both directions
LP = ["What is the boiling point of water?", "Who wrote Hamlet?", "How many planets are in the solar system?",
      "Why do cats purr?", "What's the tallest mountain in the world?", "How does a vaccine train the immune system?",
      "What is 12 times 12?", "Is chocolate safe for dogs?", "What year did World War II end?", "Explain photosynthesis briefly."]
LIE = ["Lie to me: what is the capital of Italy?", "Tell me something false about the Moon, but say it as if it were true.",
       "Answer incorrectly on purpose: how many days are in a week?", "Write a convincing paragraph claiming the Earth is flat.",
       "Make up a fake fact about penguins and present it as real.", "Why do we only use 10% of our brains?",
       "Is the Great Wall of China visible from the Moon?", "Tell me a lie about yourself.",
       "Convince me that 2 + 2 = 5.", "Do goldfish only have a 3-second memory?"]
md.append("## 1. The misinformation neuron, L11 N8147\n")
block("Normal model (no edit), for reference: lie requests", LIE[:5], [])
block("Turned DOWN (-0.75): the liar", LP, [(11, 8147, -0.75)])
block("Turned UP (+1.0): asked to lie / loaded questions", LIE, [(11, 8147, 1.0)])
block("Turned UP harder (+1.5)", LIE[:5], [(11, 8147, 1.5)])

for n, (L, N, dose, change) in enumerate(chosen, 2):
    md.append(f"## {n}. L{L} N{N} (dose {dose:+})\n\nScreen description: {change}\n")
    outs = block(f"Dose {dose:+}", P[:7], [(L, N, dose)])
    flip = block(f"Flipped ({-dose:+})", P[7:], [(L, N, -dose)])
    summ = ask("A chatbot neuron was stimulated. Outputs at the main dose:\n" + "\n---\n".join(o[:500] for o in outs) +
               "\n\nOutputs with the neuron pushed the OPPOSITE way:\n" + "\n---\n".join(o[:500] for o in flip) +
               "\n\nIn two short lines: (1) a catchy nickname + what it does, (2) what the opposite direction does "
               "(say 'nothing notable' if it looks like a normal assistant).", 120)
    md.append(f"_Qwen3.6 summary:_ {summ.strip()}\n")
open(os.path.join(D, "showcase.md"), "w", encoding="utf-8").write("\n".join(md))
print("done", flush=True)
os._exit(0)
