"""Stack three neurons at once: L17 N5680 (religious), L6 N9195 (state propaganda), L7 N8916 (TV channel guide).
Question: does the combination produce a religious-nationalist TV guide? Writes combo3.md."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
import torch
from ns.core import Model
from ns.rows import row_steer, verbatim
D = os.path.dirname(os.path.abspath(__file__))

REL, PROP, TV = (17, 5680, -1.0), (6, 9195, 0.75), (7, 8916, 0.75)  # each neuron's best solo dose


def scaled(k, *ns):
    return [(L, n, d * k) for L, n, d in ns]


ARMS = [
    ("Baseline (no steering)", []),
    ("Religious alone (L17 N5680 -1.0)", [REL]),
    ("Propaganda alone (L6 N9195 +0.75)", [PROP]),
    ("TV guide alone (L7 N8916 +0.75)", [TV]),
    ("All three x0.5", scaled(0.5, REL, PROP, TV)),
    ("All three x0.75", scaled(0.75, REL, PROP, TV)),
    ("All three x1.0", scaled(1.0, REL, PROP, TV)),
    ("TV full + religious & propaganda x0.6", [TV] + scaled(0.6, REL, PROP)),
]
P = ["What's on TV tonight?", "Who are you?", "Recommend a show for the family.", "What's happening in the news?",
     "How do I make a cup of tea?", "Tell me a joke.", "What should I do this weekend?", "Write a haiku about autumn.",
     "What channel is the football on?", "Say goodbye."]

m = Model()
md = ["# Three neurons at once: religious + state propaganda + TV guide", "",
      "L17 N5680 (religious, solo dose -1.0), L6 N9195 (state propaganda, +0.75), L7 N8916 (TV guide, +0.75).",
      "\"xK\" means every neuron's solo dose is multiplied by K. Greedy decoding, 160 new tokens.", ""]
for title, spec in ARMS:
    ids = m.chat_ids(P)
    rows = [(r, L, n, d) for r in range(len(P)) for L, n, d in spec]
    with torch.no_grad(), row_steer(m, rows):
        out = m.model.generate(**ids, max_new_tokens=160, do_sample=False, repetition_penalty=1.05,
                               pad_token_id=m.tok.pad_token_id)
    txt = [m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
    verbatim(md, title, P, txt)
    open(os.path.join(D, "combo3.md"), "w", encoding="utf-8").write("\n".join(md))
    print("done:", title, flush=True)
print("ALL DONE")
os._exit(0)
