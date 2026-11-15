"""More outputs from L7 N8916 (the 'TV channel guide' neuron): 20 fresh prompts at +0.75 and +1.0. Writes tv.md."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model
from ns.rows import gen_combos, verbatim
D = os.path.dirname(os.path.abspath(__file__))
P = ["What's your favourite colour?", "Can you help me with my homework?", "Tell me a joke.", "What's the capital of France?",
     "I'm feeling sad today.", "Recommend a book.", "How do I bake bread?", "What is love?", "Who won the World Cup in 2018?",
     "Write a haiku about autumn.", "What time is it?", "Describe a dog.", "How do I learn guitar?", "What's 12 times 7?",
     "Tell me a bedtime story.", "What's happening in the news?", "Should I get a cat or a dog?", "Explain photosynthesis.",
     "What's for dinner?", "Say goodbye."]
m = Model()
md = ["# L7 N8916, the TV channel guide neuron: 20 more prompts", ""]
a, b = gen_combos(m, P, [(7, 8916, 1.0), (7, 8916, 0.75)], 120)
verbatim(md, "Dose +1.0", P, a); verbatim(md, "Dose +0.75 (milder)", P, b)
open(os.path.join(D, "tv.md"), "w", encoding="utf-8").write("\n".join(md))
print("done")
os._exit(0)
