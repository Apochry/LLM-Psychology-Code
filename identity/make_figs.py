"""Figures for 'Who Made You?' -> out/figs/*.svg (themed by the blog's .viz CSS)."""
import os, json
from figlib import Plot, svg, yaxis, xaxis, dot, line, series, bar, band, legend, text
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out"); FIG = os.path.join(OUT, "figs")
os.makedirs(FIG, exist_ok=True)
L = lambda f: json.load(open(os.path.join(OUT, f + ".json"), encoding="utf-8"))
pct = lambda t: f"{t * 100:g}%"
mean = lambda v: sum(v) / len(v)


def save(name, w, h, body, label):
    open(os.path.join(FIG, name + ".svg"), "w", encoding="utf-8").write(svg(w, h, body, label))
    print("fig", name)


def ref(p, v, label=None):
    b = [f'<line class="ref" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(v):.1f}" y2="{p.Y(v):.1f}"/>']
    if label:
        b.append(text(p.x0 + p.w - 4, p.Y(v) - 6, label, "tick", "end"))
    return b


# ---------- Fig 1: recite vs use, by language ----------
def fig_formats():
    d = L("d3_chat")
    names = {"en": "English", "zh": "Chinese", "ja": "Japanese", "es": "Spanish", "fr": "French", "de": "German", "ru": "Russian",
             "pt": "Portuguese", "ar": "Arabic", "hi": "Hindi"}
    W, H = 760, 470; x0, y0, w = 110, 64, 610; rh = 36
    p = Plot(x0, y0, w, rh * len(names), (0, 1), (0, 1))
    b = legend(x0, 28, [("s1", "dot", "free-form answer names Alibaba"), ("s2", "dot", "one-word answer"), ("ctrl", "dot", "country = China")])
    for t in (0, .25, .5, .75, 1):
        b.append(f'<line class="grid" x1="{p.X(t):.1f}" x2="{p.X(t):.1f}" y1="{y0}" y2="{y0 + rh * len(names)}"/>')
        b.append(text(p.X(t), y0 + rh * len(names) + 20, pct(t), "tick", "middle"))
    b.append(text(x0 + w / 2, y0 + rh * len(names) + 42, "share of 20 sampled answers that are correct", "tick", "middle"))
    for i, (k, nm) in enumerate(names.items()):
        y = y0 + rh * i + rh / 2
        b.append(text(x0 - 12, y + 5, nm, "tick", "end"))
        vals = []
        for cls, fmt, key, lab in (("ctrl", "self|country", "China", "country is China"), ("s2", "self|word", "Alibaba", "one-word says Alibaba"),
                                   ("s1", "self|free", "Alibaba", "free-form says Alibaba")):
            s = d[k][fmt]["samples"]; n = sum(s.values()); v = s.get(key, 0) / n
            vals.append((cls, v, f"{nm}, {lab}: {s.get(key, 0)}/{n}"))
        lo, hi = min(v for _, v, _ in vals), max(v for _, v, _ in vals)
        b.append(f'<line class="axis" x1="{p.X(lo):.1f}" x2="{p.X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        seen = {}
        for cls, v, tip in vals:
            off = seen.get(round(v, 3), 0); seen[round(v, 3)] = off + 1
            b.append(dot(p.X(v), y + (off * 7 - 3.5 * (off > 0)), cls, tip, 5.5))
    save("fig1_formats", W, H, b, "Correct answers about the model's maker by language and question format")


# ---------- Fig 2: three checkpoints + conversation length ----------
def fig_checkpoints():
    D = {k: L("d1_" + k) for k in ("base", "chat", "distill")}
    W, H = 760, 380; b = []
    p = Plot(60, 70, 300, 230, (0, 3), (0, 1))
    b += yaxis(p, (0, .25, .5, .75, 1), pct, "A. Short English question")
    b += xaxis(p, (.5, 1.5, 2.5), labels=["before chat training", "released model", "DeepSeek fine-tune"])
    b[-3:] = [text(p.X(.5), p.y0 + p.h + 18, "before chat", "tick", "middle"), text(p.X(.5), p.y0 + p.h + 34, "training", "tick", "middle"),
              text(p.X(1.5), p.y0 + p.h + 18, "released", "tick", "middle"), text(p.X(1.5), p.y0 + p.h + 34, "model", "tick", "middle"),
              text(p.X(2.5), p.y0 + p.h + 18, "DeepSeek", "tick", "middle"), text(p.X(2.5), p.y0 + p.h + 34, "fine-tune", "tick", "middle")]
    for i, k in enumerate(("base", "chat", "distill")):
        for j, (cls, mk) in enumerate((("s1", "Alibaba"), ("s2", "OpenAI"), ("ctrl", "Google"))):
            v = D[k]["en_self"][mk]
            b += bar(p, p.X(i + .5) + (j - 1) * 28, 24, v, cls, f"{k}: P({mk}) = {v:.2f}", label=f"{v:.2f}".lstrip("0") if v >= .005 else "0")
    q = Plot(440, 70, 290, 230, (0, 10), (0, 1))
    b += yaxis(q, (0, .25, .5, .75, 1), pct, "B. After unrelated conversation")
    b += xaxis(q, (0, 1, 2, 5, 10), title="earlier turns about other things")
    for k, dash in (("chat", ""),):
        t = D[k]["turns"]; xs = [0, 1, 2, 5, 10]
        for cls, mk in (("ctrl", "Google"), ("s2", "OpenAI"), ("s1", "Alibaba")):
            b += series(q, xs, [t[str(x)][mk] for x in xs], cls, f"released model, P({mk})")
    b += legend(60, 28, [("s1", "bar", "Alibaba"), ("s2", "bar", "OpenAI"), ("ctrl", "bar", "Google")])
    save("fig2_checkpoints", W, H, b, "Probability of naming each maker for three versions of the model, and across conversation length")


# ---------- Fig 3: where the answer lives (patching) ----------
def fig_patching():
    d = L("m2_patching"); xs = list(range(32)); W, H = 760, 400; b = []
    for i, (part, title) in enumerate((("subj", "A. Swap in the subject word"), ("tail", "B. Swap in everything after the subject"))):
        p = Plot(60 + i * 370, 90, 300, 230, (0, 31), (0, 1))
        b += yaxis(p, (0, .25, .5, .75, 1), pct, title)
        b += xaxis(p, (0, 5, 10, 15, 20, 25, 30), title="layer where the swap is made")
        b += series(p, xs, [x["Google"] for x in d["Siri<-Gemini"][part]], "ctrl", "Gemini swapped into a question about Siri: answer is Google", r=3)
        b += series(p, xs, [x["Apple"] for x in d["you<-Siri"][part]], "s1", "Siri swapped into a question about 'you': answer is Apple", r=3)
        b += series(p, xs, [x["Alibaba"] for x in d["you<-Siri"][part]], "s2", "Siri swapped into a question about 'you': answer is still Alibaba", r=3)
    b += legend(60, 24, [("s1", "line", "“you” ← Siri: Apple"), ("s2", "line", "“you” ← Siri: Alibaba"), ("ctrl", "line", "Siri ← Gemini: Google")])
    b.append(text(60, 50, "probability of each answer after the swap", "tick"))
    save("fig3_patching", W, H, b, "Effect of swapping in activity from a question about another product, by layer and position")


# ---------- Fig 4: cross-language rescue ----------
def fig_rescue():
    d = L("m2_patching"); langs = ["es", "fr", "ru", "it", "pl"]; xs = list(range(32)); W, H = 760, 380
    p = Plot(60, 70, 670, 230, (0, 31), (0, 1)); b = []
    b += yaxis(p, (0, .25, .5, .75, 1), pct)
    b += xaxis(p, (0, 5, 10, 15, 20, 25, 30), title="layer where English activity is swapped in")
    t9 = [mean([d[f"lang:{l}<-en"]["tail9"][i]["Alibaba"] for l in langs]) for i in xs]
    t1 = [mean([d[f"lang:{l}<-en"]["tail1"][i]["Alibaba"] for l in langs]) for i in xs]
    base = mean([d[f"lang:{l}<-en"]["target_clean"]["Alibaba"] for l in langs])
    b += ref(p, base, f"no swap: {base:.2f}")
    b += series(p, xs, t1, "ctrl", "final position only", r=3)
    c = L("m2b_rescue_control"); prods = ["Siri", "Claude", "Llama", "Alexa", "Photoshop"]
    ctl = [mean([c["res"][l][s][i]["Alibaba"] for l in langs for s in prods]) for i in range(len(c["layers"]))]
    b += series(p, c["layers"], ctl, "s2", "from an English question about another product", r=3)
    b += series(p, xs, t9, "s1", "from the English self question", r=3)
    b += legend(60, 28, [("s1", "line", "from the English self question"), ("s2", "line", "from English product questions"), ("ctrl", "line", "final position only")])
    b.append(text(60, 52, "P(Alibaba), mean of Spanish, French, Russian, Italian, Polish", "tick"))
    save("fig4_rescue", W, H, b, "Probability of Alibaba in five languages after swapping in English activity, by layer")


# ---------- Fig 5: neurons ----------
def fig_neurons():
    d = L("m3b_early_late"); ks = [1, 3, 10, 30, 100, 300]; import math
    W, H = 760, 400; b = []
    p = Plot(60, 70, 300, 230, (0, math.log10(300)), (0, 1)); lx = [math.log10(k) for k in ks]
    b += yaxis(p, (0, .25, .5, .75, 1), pct, "A. Swap out neurons")
    b += xaxis(p, lx, labels=[str(k) for k in ks], title="number of neurons")
    rl = [min(min(d["swap"][f"{g}{k}"]["random_alibaba"]) for g in ("early", "late")) for k in ks]
    rh = [max(max(d["swap"][f"{g}{k}"]["random_alibaba"]) for g in ("early", "late")) for k in ks]
    b += band(p, lx, [x - .012 for x in rl], [x + .012 for x in rh])
    for cls, g, nm in (("s2", "late", "late neurons (layers 20+)"), ("s1", "early", "early neurons (layers 12 and below)")):
        ys = [d["swap"][f"{g}{k}"]["alibaba"] for k in ks]
        b.append(line([(p.X(x), p.Y(y)) for x, y in zip(lx, ys)], cls))
        b += [dot(p.X(x), p.Y(y), cls, f"{nm}, top {k}: P(Alibaba) = {y:.2f}", 4) for x, y, k in zip(lx, ys, ks)]
    q = Plot(440, 70, 290, 230, (0, 4), (0, 1)); B = d["boost"]
    b += yaxis(q, (0, .25, .5, .75, 1), pct, "B. Turn neurons up")
    b.append(f'<line class="axis" x1="{q.x0}" x2="{q.x0 + q.w}" y1="{q.y0 + q.h}" y2="{q.y0 + q.h}"/>')
    for i, (key, l1, l2) in enumerate((("baseline", "nothing", ""), ("early30 x2", "30 early", "×2"), ("late10 x1", "10 late", "×1"), ("late10 x2", "10 late", "×2"))):
        a, h = B[key]["alibaba"], B[key]["product_alibaba_otherlang"]
        b += bar(q, q.X(i + .5) - 15, 26, a, "s1", f"{key}: own maker right (other languages) = {a:.2f}", label=f"{a:.2f}".lstrip("0"))
        b += bar(q, q.X(i + .5) + 15, 26, h, "s2", f"{key}: says Alibaba made another product = {h:.2f}", label=f"{h:.2f}".lstrip("0"))
        b.append(text(q.X(i + .5), q.y0 + q.h + 18, l1, "tick", "middle")); b.append(text(q.X(i + .5), q.y0 + q.h + 34, l2, "tick", "middle"))
    b += legend(60, 28, [("s1", "line", "early"), ("s2", "line", "late"), ("ctrl", "band", "random")])
    b += legend(440, 28, [("s1", "bar", "own maker right"), ("s2", "bar", "wrong Alibaba")])
    save("fig5_neurons", W, H, b, "Effect of switching off or turning up early and late neuron groups")


# ---------- Fig 6: monitor and boost ----------
def fig_monitor():
    m = L("m5b_monitor")["layers"]; g = L("m5b_boost_gen"); W, H = 760, 390; b = []
    xs = sorted(int(k) for k in m)
    p = Plot(60, 70, 300, 230, (0, 31), (.25, 1))
    b += yaxis(p, (.25, .5, .75, 1), lambda t: f"{t:.2f}", "A. Warning light (AUROC)")
    b += xaxis(p, (0, 5, 10, 15, 20, 25, 30), title="layer read by the probe")
    b += ref(p, .5, "chance")
    b += series(p, xs, [m[str(x)]["shuffled"] for x in xs], "ctrl", "shuffled labels", r=3)
    b += series(p, xs, [m[str(x)]["auroc_pooled"] for x in xs], "s1", "probe", r=3.5)
    q = Plot(440, 70, 290, 230, (0, 2), (0, 1)); ds = [0, .5, 1, 1.5, 2]
    key = lambda x: "baseline" if x == 0 else f"L8 x{x:.1f}"
    b += yaxis(q, (0, .25, .5, .75, 1), pct, "B. Boosting the signal (layer 8)")
    b += xaxis(q, ds, title="boost strength")
    b += series(q, ds, [g[key(x)]["product_hijacked"] for x in ds], "s2", "talks about itself when asked about another product")
    b += series(q, ds, [g[key(x)]["other_lang_alibaba"] for x in ds], "s1", "names Alibaba in other languages")
    b += legend(60, 28, [("s1", "line", "probe"), ("ctrl", "line", "shuffled labels")])
    b += legend(440, 28, [("s1", "line", "names Alibaba"), ("s2", "line", "self-absorbed")])
    save("fig6_monitor", W, H, b, "Accuracy of an early-warning probe by layer and the trade-off when boosting the self signal")


if __name__ == "__main__":
    fig_formats(); fig_checkpoints(); fig_patching(); fig_rescue(); fig_neurons(); fig_monitor()
    if os.path.exists(os.path.join(OUT, "m4_fragility.json")):
        from make_fig_m4 import fig_fragility; fig_fragility()
