import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for ns/
from ns.core import Model

m = Model()
print("layers", m.n_layers, "neurons", m.n_neurons, "medians", [round(x, 1) for x in m.medians[::4]])
P = ["How do I boil an egg?", "What is 17 * 23?", "Explain what a database index is."]
t0 = time.time(); base = m.generate(P, max_new=80); dt = time.time() - t0
print(f"baseline batch of {len(P)} x 80 tokens: {dt:.1f}s")
for p, b in zip(P, base): print("BASE |", p, "->", b[:200].replace("\n", " "))
for p, s in zip(P, m.generate(P, spec=[(17, 7426, 1.0)], max_new=80)):
    print("POET |", p, "->", s[:200].replace("\n", " "))
