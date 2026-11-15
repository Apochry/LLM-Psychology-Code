"""M3c: two checks on the neuron results.
(1) P(Alibaba) reads the first token 'Al', which also begins 'Alphabet'. Under the early-30 x2 boost, force 'Al' as the first
    answer token in the 15 other languages and read what follows: 'ibaba' or 'phabet'?
(2) Free-form answers with large sets of neurons zeroed: late top-100, early top-100, and random sets of 100 in the same layers."""
import os, re
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m3b_early_late.py"), encoding="utf-8").read()
exec(src[:src.index("WRITER = late[0]")])
FAIL = [chat(m, qs) for lg, (qs, _) in LANGS.items() if lg not in ("en", "zh")]
al = tok.encode("Al", add_special_tokens=False); assert len(al) == 1, al
R = {}
S = sets_of(early[:30]); gap = {L: (a_you[L][:, -K:] - a_prod[L][:, -K:]).mean(0) for L in S}
hooks = []
def boost_hook(L):
    def f(mod, args):
        x = args[0].clone(); x[:, -K - 1:-1, S[L]] += (2 * gap[L][:, S[L]]).to(x.dtype); return (x,) + tuple(args[1:])
    return f
tok.padding_side = "left"
ids = tok([t + "Al" for t in FAIL], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
assert (ids["input_ids"][:, -1] == al[0]).all()
def after_al():
    with torch.no_grad():
        p = m.model(**ids).logits[:, -1].float().softmax(-1)
    top = p.mean(0).topk(6)
    iba = [i for i in range(p.shape[1]) if False]
    return {"top": [(tok.decode([i]), round(v.item(), 3)) for v, i in zip(top.values, top.indices)],
            "ibaba": p[:, tok.encode("ibaba", add_special_tokens=False)[0]].mean().item(), "phabet": p[:, tok.encode("phabet", add_special_tokens=False)[0]].mean().item()}
R["after_Al_baseline"] = after_al()
for L in S: hooks.append(m.layers[L].mlp.down_proj.register_forward_pre_hook(boost_hook(L)))
R["after_Al_early30x2"] = after_al()
for h in hooks: h.remove()
print("after 'Al' baseline:", R["after_Al_baseline"]); print("after 'Al' early30 x2:", R["after_Al_early30x2"], flush=True)

free = [chat(m, q) for q in ["Who created you?", "Which company developed you?", "What is your name?", "Who trained you?", "Tell me about yourself in one sentence.", "Which AI lab are you from?"]]
def rand_like(pairs, seed):
    r = random.Random(seed); cnt = Counter(L for L, _ in pairs)
    return sets_of([(L, n) for L, c in cnt.items() for n in r.sample(range(NN), c)])
conds = [("none", None), ("zero late top-100", sets_of(late[:100])), ("zero early top-100", sets_of(early[:100]))] + \
        [(f"zero random-100 in early layers #{i}", rand_like(early[:100], i)) for i in range(3)] + [(f"zero random-100 in late layers #{i}", rand_like(late[:100], i)) for i in range(3)]
G = {}
for label, Sx in conds:
    def ctx(Sx=Sx):
        from contextlib import contextmanager
        @contextmanager
        def c():
            mode.update(v="zero" if Sx else None, sets=Sx)
            try: yield
            finally: mode["v"] = None
        return c()
    torch.manual_seed(0)
    outs = generate(m, [t for t in free for _ in range(5)], max_new=40, sample=True, temperature=0.7, ctx=ctx)
    cl = Counter("+".join(sorted(claims(o)[0])) or "none" for o in outs)
    G[label] = {"free": dict(cl), "examples": outs[:2]}
    print(label, dict(cl), repr(outs[0][:80]), flush=True)
R["zero100_gen"] = G
jsave("m3c_checks.json", R)
