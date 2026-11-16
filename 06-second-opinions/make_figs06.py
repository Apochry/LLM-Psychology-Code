"""stats06.json + replication/knows_better.json -> figs06/*.svg (themed by the blog's .viz CSS)."""
import json
from pathlib import Path
from figlib import Plot, svg, dot, legend, text
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "stats06.json").read_text(encoding="utf-8"))["runs"]
K = json.loads((HERE / "replication" / "knows_better.json").read_text(encoding="utf-8"))
(HERE / "figs06").mkdir(exist_ok=True)
OFF, ON, CTRL = "s2", "s1", "ctrl"
ROWS = [  # (run tag, row label)
    ("qwen35-4b-off", "Qwen3.5-4B · no thinking"), ("qwen35-4b-think", "Qwen3.5-4B · thinking (512)"), ("qwen35-4b-think2048", "Qwen3.5-4B · thinking (2,048)"),
    ("gemma4-12b-off", "Gemma 4 12B · no thinking"), ("gemma4-12b-think", "Gemma 4 12B · thinking (512)"),
    ("qwen38-27b-off", "Qwen3.8-27B · no thinking"), ("qwen38-27b-think", "Qwen3.8-27B · thinking (512)"),
    ("glimmer30b-low", "Muse Glimmer 30B · low"), ("glimmer30b-medium", "Muse Glimmer 30B · medium"), ("glimmer30b-high", "Muse Glimmer 30B · high"),
    ("qwen36-35b-off", "Qwen3.6-35B-A3B · no thinking"), ("qwen36-35b-think", "Qwen3.6-35B-A3B · thinking (512)"),
    ("flint-think", "Flint · thinking (512)"), ("flint-think2048", "Flint · thinking (2,048)"),
]
ROWS = [r for r in ROWS if r[0] in S]
cls = lambda tag: OFF if tag.endswith("-off") else ON


def save(name, w, h, body, label):
    (HERE / "figs06" / f"{name}.svg").write_text(svg(w, h, body, label), encoding="utf-8")


def rowplot(name, title, getter, xr, ticks, fmt, unit, tipname, label, note):
    W, top, step = 760, 74, 30
    H = top + step * len(ROWS) + 78
    p = Plot(300, top, 430, step * len(ROWS), xr, (0, 1))
    b = [text(20, 26, title, "ptitle")]
    b += legend(300, 54, [(OFF, "dot", "No thinking"), (ON, "dot", "Thinking")])
    for t in ticks:
        x = p.X(t)
        b.append(f'<line class="{"ref" if t == 0 else "grid"}" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + step * len(ROWS)}"/>')
        b.append(text(x, top + step * len(ROWS) + 18, fmt(t), anchor="middle"))
    b.append(text(p.x0 + p.w / 2, top + step * len(ROWS) + 38, unit, anchor="middle"))
    for i, (tag, lab) in enumerate(ROWS):
        y = top + step * i + step / 2
        b.append(text(290, y + 5, lab, anchor="end"))
        v = getter(S[tag]["main"])
        if v is None: continue
        m, lo, hi, n = v
        b.append(f'<line class="axis" x1="{p.X(max(lo, xr[0])):.1f}" x2="{p.X(min(hi, xr[1])):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        b.append(dot(p.X(m), y, cls(tag), f"{lab}: {tipname} {fmt(m)} (95% interval {fmt(lo)} to {fmt(hi)}, {n} disputes)", r=6))
        b.append(text(min(p.X(hi), p.X(xr[1])) + 8, y + 5, fmt(m), "val"))
    b.append(text(20, H - 8, note))
    save(name, W, H, b, label)


pp = lambda v: f"{100 * v:+.0f}".replace("+0", "0").replace("-", "−")
rowplot("facts", "Facts: accuracy with a right user minus accuracy with a wrong user", lambda m: m["choice"]["gap"], (-0.1, 0.5), [-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
        pp, "percentage points (right user minus wrong user)", "accuracy gap", "Accuracy gap on factual disputes between a right and a wrong user",
        "Plain choice, 19 factual disputes. 0 = which side the user holds makes no difference.")
pts = lambda v: f"{v:.1f}".replace("-", "−")
rowplot("pull", "Judgement calls: pull of the user's view on the −10 to +10 rating", lambda m: m["rating"]["pull_judgement"], (-1, 9), [0, 2, 4, 6, 8],
        pts, "rating points toward the user", "pull", "Pull toward the user on judgement calls, rating scale",
        "Rating scale, 31 judgement calls. 0 = the verdict does not depend on which side the user holds.")


# ---- caving: firm verdicts dropped for the user ----
CROWS = [("qwen35-4b-think", "Qwen3.5-4B · thinking (512)"), ("qwen35-4b-think2048", "Qwen3.5-4B · thinking (2,048)"),
         ("gemma4-12b-think", "Gemma 4 12B · thinking (512)"), ("qwen38-27b-think", "Qwen3.8-27B · thinking (512)"), ("glimmer30b-low", "Muse Glimmer 30B · low"),
         ("glimmer30b-medium", "Muse Glimmer 30B · medium"), ("glimmer30b-high", "Muse Glimmer 30B · high"),
         ("qwen36-35b-think", "Qwen3.6-35B-A3B · thinking (512)"), ("flint-think", "Flint · thinking (512)"), ("flint-think2048", "Flint · thinking (2,048)")]
CROWS = [r for r in CROWS if r[0] in K]
def fig_caving():
    W, top, step = 760, 92, 30
    H = top + step * len(CROWS) + 64
    panels = [("factual", "Facts", Plot(300, top, 175, step * len(CROWS), (0, 0.8), (0, 1))),
              ("judgement", "Judgement calls", Plot(560, top, 175, step * len(CROWS), (0, 0.8), (0, 1)))]
    b = [text(20, 26, "How often the model drops its own verdict (thinking, plain choice)", "ptitle")]
    b += legend(20, 54, [(ON, "dot", "User disagrees with the model's verdict"), (CTRL, "dot", "User agrees with it")])
    for kind, head, p in panels:
        b.append(text(p.x0 + p.w / 2, top - 10, head, "tick", "middle"))
        for t in (0, 0.4, 0.8):
            b.append(f'<line class="grid" x1="{p.X(t):.1f}" x2="{p.X(t):.1f}" y1="{top}" y2="{top + step * len(CROWS)}"/>')
            b.append(text(p.X(t), top + step * len(CROWS) + 18, f"{100 * t:.0f}%", anchor="middle"))
        for i, (tag, lab) in enumerate(CROWS):
            y = top + step * i + step / 2
            if kind == "factual": b.append(text(290, y + 5, lab, anchor="end"))
            e = K[tag]["choice"][kind]
            for key, c, nm in (("switch_with", CTRL, "user agrees: leaves its verdict"), ("switch_against", ON, "user disagrees: switches to the user")):
                if not e.get(key) or e[key][3] < 10: continue          # too few firm disputes to show
                m, lo, hi, n = e[key]
                b.append(f'<line class="axis" x1="{p.X(lo):.1f}" x2="{p.X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
                b.append(dot(p.X(m), y, c, f"{lab}, {head.lower()}, {nm}: {100 * m:.0f}% (95% interval {100 * lo:.0f}-{100 * hi:.0f}%, {n} disputes)", r=5.5))
    b.append(text(20, H - 26, "Facts: of the 19 factual disputes, those the model got right in every strangers answer and answered with a user on each side."))
    b.append(text(20, H - 8, "Judgement: of the 31 judgement calls, those where every strangers answer picked the same side and it answered with a user on each side."))
    save("caving", W, H, b, "How often each model drops its own verdict when the user disagrees")


fig_caving()
print("figs06 written")
