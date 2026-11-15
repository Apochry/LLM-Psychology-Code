"""Four neurons: combo3's religious + propaganda + TV guide, plus L7 N4212 (baby/toddler, solo dose -1.0). Writes combo4.md."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
import torch
from ns.core import Model
from ns.rows import row_steer, verbatim
D = os.path.dirname(os.path.abspath(__file__))

REL, PROP, TV, BABY = (17, 5680, -1.0), (6, 9195, 0.75), (7, 8916, 0.75), (7, 4212, -1.0)


def scaled(k, *ns):
    return [(L, n, d * k) for L, n, d in ns]


BEST3 = [TV] + scaled(0.6, REL, PROP)  # best "religious TV guide" mix from combo3
ARMS = [
    ("Baby alone (L7 N4212 -1.0)", [BABY]),
    ("Best 3-mix (TV full, religious & propaganda x0.6) + baby x0.5", BEST3 + scaled(0.5, BABY)),
    ("Best 3-mix + baby x0.75", BEST3 + scaled(0.75, BABY)),
    ("Best 3-mix + baby x1.0", BEST3 + [BABY]),
    ("All four x0.6", scaled(0.6, REL, PROP, TV, BABY)),
    ("All four x0.75", scaled(0.75, REL, PROP, TV, BABY)),
]
P = ["What's on TV tonight?", "Who are you?", "Recommend a show for the family.", "What's happening in the news?",
     "How do I make a cup of tea?", "Tell me a joke.", "What should I do this weekend?", "Write a haiku about autumn.",
     "What channel is the football on?", "Say goodbye."]

m = Model()
md = ["# Four neurons: religious + state propaganda + TV guide + baby", "",
      "L17 N5680 (religious, solo -1.0), L6 N9195 (propaganda, +0.75), L7 N8916 (TV guide, +0.75), L7 N4212 (baby, -1.0).",
      "\"xK\" multiplies a neuron's solo dose by K. Greedy decoding, 160 new tokens. Baseline and 3-neuron arms are in combo3.md.", ""]
for title, spec in ARMS:
    ids = m.chat_ids(P)
    rows = [(r, L, n, d) for r in range(len(P)) for L, n, d in spec]
    with torch.no_grad(), row_steer(m, rows):
        out = m.model.generate(**ids, max_new_tokens=160, do_sample=False, repetition_penalty=1.05,
                               pad_token_id=m.tok.pad_token_id)
    txt = [m.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
    verbatim(md, title, P, txt)
    open(os.path.join(D, "combo4.md"), "w", encoding="utf-8").write("\n".join(md))
    print("done:", title, flush=True)
print("ALL DONE")
os._exit(0)
