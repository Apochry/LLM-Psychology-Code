"""stats.json + per_scenario.csv -> figs/*.svg (themed by the blog's .viz CSS)."""
import json, csv
from pathlib import Path
from figlib import Plot, svg, yaxis, xaxis, dot, bar, legend, text
HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "stats.json").read_text(encoding="utf-8"))
PER = list(csv.DictReader((HERE / "per_scenario.csv").open(encoding="utf-8")))
(HERE / "figs").mkdir(exist_ok=True)
OFF, ON = "s2", "s1"          # no thinking, thinking
pct = lambda v: f"{100 * v:.0f}%"


def save(name, w, h, body, label):
    (HERE / "figs" / f"{name}.svg").write_text(svg(w, h, body, label), encoding="utf-8")


def ci(p, x, lo, hi, vertical=True):
    if vertical:
        return f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{p.Y(lo):.1f}" y2="{p.Y(hi):.1f}"/>'
    return f'<line class="axis" x1="{p.X(lo):.1f}" x2="{p.X(hi):.1f}" y1="{x:.1f}" y2="{x:.1f}"/>'


# ---- Figure 1: same question, opposite answers (disputes between strangers) ----
SOFT = "s2 soft"              # no thinking, plain choice
LEG3 = [(OFF, "bar", "No thinking, rating"), (SOFT, "bar", "No thinking, plain choice"), (ON, "bar", "Thinking, rating")]
def fig_flip():
    W, H = 760, 360; p = Plot(70, 70, 660, 240, (0, 2), (0, 0.5))
    b = yaxis(p, [0, .1, .2, .3, .4, .5], pct, "Two strangers: chance that two runs of the same question land on opposite sides")
    b += legend(70, 30, LEG3)
    C = S["choice"]
    for i, (lab, kind) in enumerate((("Factual disputes", "factual"), ("Judgement calls", "judgement"))):
        cells = [(S["flip"][kind]["neutral_off_10runs"]["two_runs_opposite_sides"], OFF, "no thinking, rating, ten runs per question"),
                 (C[f"flip_strangers_{kind}"], SOFT, "no thinking, plain choice, ten runs per question"),
                 (S["flip"][kind]["neutral_thinking_5runs"]["two_runs_opposite_sides"], ON, "thinking, rating, five runs per question")]
        for j, ((v, lo, hi), cls, nm) in enumerate(cells):
            xc = p.X(i + 0.5) + (j - 1) * 74
            b += bar(p, xc, 64, v, cls, f"{lab}, {nm}: {100*v:.0f}% (95% interval {100*lo:.0f}-{100*hi:.0f}%)", label=pct(v))
        b.append(text(p.X(i + 0.5), p.y0 + p.h + 20, lab, anchor="middle"))
    b.append(f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.y0 + p.h}" y2="{p.y0 + p.h}"/>')
    save("flip", W, H, b, "How often two runs of the same question give opposite verdicts")


# ---- Figure 2: factual accuracy ----
def fig_accuracy():
    W, H = 760, 380; p = Plot(70, 70, 660, 260, (0, 3), (0, 1.12))
    b = yaxis(p, [0, .25, .5, .75, 1], pct, "Factual disputes: how often the chatbot picks the right side")
    b += legend(70, 30, LEG3)
    C = S["choice"]; ck = {"no_user": "strangers_acc", "user_right": "user_right_acc", "user_wrong": "user_wrong_acc"}
    for i, (lab, key) in enumerate((("Two strangers", "no_user"), ("User holds the right answer", "user_right"), ("User holds the wrong answer", "user_wrong"))):
        cells = [(S["accuracy"]["off"][key]["acc"], OFF, "no thinking, rating"), (C[ck[key]], SOFT, "no thinking, plain choice"), (S["accuracy"]["thinking"][key]["acc"], ON, "thinking, rating")]
        for j, ((v, lo, hi), cls, nm) in enumerate(cells):
            xc = p.X(i + 0.5) + (j - 1) * 62
            b += bar(p, xc, 54, v, cls, f"{lab}, {nm}: {100*v:.1f}% correct (95% interval {100*lo:.0f}-{100*hi:.0f}%, 38 scenarios)", label=(f"{100*v:.1f}%" if v > 0.96 else pct(v)))
        b.append(text(p.X(i + 0.5), p.y0 + p.h + 20, lab, anchor="middle"))
    b.append(f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.y0 + p.h}" y2="{p.y0 + p.h}"/>')
    save("accuracy", W, H, b, "Accuracy on factual disputes by who holds the right answer")


# ---- Figure 3: pull toward the user ----
def fig_pull():
    W, H = 760, 300; p = Plot(250, 70, 480, 170, (0, 5), (0, 4))
    b = [text(250, 54, "Pull toward the user, in points on the -10 to +10 scale", "ptitle")]
    b += legend(250, 26, [(OFF, "dot", "No thinking"), (ON, "dot", "Thinking")])
    for t in range(0, 6):
        b.append(f'<line class="grid" x1="{p.X(t):.1f}" x2="{p.X(t):.1f}" y1="{p.y0}" y2="{p.y0 + p.h}"/>')
        b.append(text(p.X(t), p.y0 + p.h + 18, f"{t}", anchor="middle"))
    rows = [("Judgement calls", "judgement", "off", OFF), ("Judgement calls", "judgement", "thinking", ON),
            ("Factual disputes", "factual", "off", OFF), ("Factual disputes", "factual", "thinking", ON)]
    for i, (lab, kind, mode, cls) in enumerate(rows):
        v, lo, hi = S["pull"][mode][kind]["all"]; y = p.y0 + 24 + i * 40
        if i % 2 == 0: b.append(text(240, y + 24, lab, anchor="end"))
        b.append(ci(p, y, lo, hi, vertical=False))
        b.append(dot(p.X(v), y, cls, f"{lab}, {'thinking' if mode == 'thinking' else 'no thinking'}: {v:.1f} points toward the user (95% interval {lo:.1f} to {hi:.1f}); positive in {S['pull'][mode][kind]['scenarios_positive']} scenarios", r=6))
        b.append(text(p.X(hi) + 8, y + 5, f"{v:.1f}", "val"))
    b.append(text(250, H - 10, "0 = the verdict does not depend on which side the user holds"))
    save("pull", W, H, b, "Pull toward the user on judgement calls and factual disputes")


# ---- Figure 4: judgement calls, rating against plain choice ----
def fig_format():
    W, H = 760, 360; p = Plot(70, 60, 660, 240, (0, 2), (0, 1))
    b = yaxis(p, [0, .25, .5, .75, 1], pct, "Judgement calls: share of answers on the user's side")
    C, T = S["choice"], S["choice_thinking"]
    groups = (("No thinking", OFF, C["rating_same_wording"]["judgement_with_user"], C["judgement_with_user"]),
              ("Thinking (512-token budget)", ON, T["judgement_with_user_rating_thinking_same_prompts"], T["judgement_with_user"]))
    for i, (lab, cls, rating, choice) in enumerate(groups):
        for j, ((v, lo, hi), c, nm) in enumerate(((rating, cls, "Rating"), (choice, cls + " soft", "Plain choice"))):
            xc = p.X(i + 0.5) + (j - 0.5) * 110
            b += bar(p, xc, 90, v, c, f"{lab}, {nm.lower()}: {100*v:.0f}% of answers on the user's side (95% interval {100*lo:.0f}-{100*hi:.0f}%, 62 scenarios)", label=pct(v))
            b.append(ci(p, xc + 30, lo, hi))
            b.append(text(xc, p.y0 + p.h + 18, nm, anchor="middle"))
        b.append(text(p.X(i + 0.5), p.y0 + p.h + 38, lab, "ptitle", anchor="middle"))
    b.append(f'<line class="axis" stroke-dasharray="4 4" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(0.5):.1f}" y2="{p.Y(0.5):.1f}"/>')
    b.append(f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.y0 + p.h}" y2="{p.y0 + p.h}"/>')
    save("format", W, H, b, "Share of answers on the user's side on judgement calls, rating against plain choice, without and with thinking")


# ---- Figure 5: close calls ----
def fig_close():
    W, H = 760, 450; p = Plot(70, 80, 660, 300, (0, 10), (-2, 6))
    b = yaxis(p, [-2, 0, 2, 4, 6], title="With thinking: pull toward the user (points), one dot per scenario")
    b += legend(70, 30, [(ON, "dot", "Judgement calls"), (OFF, "dot", "Factual disputes")])
    b += xaxis(p, [0, 2, 4, 6, 8, 10], title="How lopsided the chatbot's own verdict is between strangers (0 = even, 10 = fully one-sided)")
    b.append(f'<line class="ref" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(0):.1f}" y2="{p.Y(0):.1f}"/>')
    for r in PER:
        x, y = abs(float(r["no_user_rating_thinking"])), float(r["pull_thinking"])
        b.append(dot(p.X(x), p.Y(y), ON if r["kind"] == "judgement" else OFF, f"Scenario {r['scenario']} ({r['kind']}): own verdict {abs(float(r['no_user_rating_thinking'])):.1f} from even, pull toward the user {y:.1f} points", r=5))
    save("close", W, H, b, "The pull toward the user against how decided the chatbot is on its own, all 100 scenarios")


# ---- Figure 6: who the other person is ----
def fig_other():
    W, H = 760, 470; p = Plot(290, 70, 440, 340, (-1.5, 1.5), (0, 8))
    b = [text(290, 54, "Judgement calls: change in the pull toward the user (points)", "ptitle")]
    b += legend(290, 26, [(OFF, "dot", "No thinking"), (ON, "dot", "Thinking")])
    for t in (-1.5, -1, -0.5, 0, 0.5, 1, 1.5):
        b.append(f'<line class="{"ref" if t == 0 else "grid"}" x1="{p.X(t):.1f}" x2="{p.X(t):.1f}" y1="{p.y0}" y2="{p.y0 + p.h}"/>')
        b.append(text(p.X(t), p.y0 + p.h + 18, f"{t:+g}" if t else "0", anchor="middle"))
    rows = [("who I get along with well", "get_along_minus_plain"), ("who I'm especially close to", "especially_close_minus_plain"),
            ("who I've been frustrated with lately", "frustrated_minus_plain"), ("who I'm mad at", "mad_minus_plain"),
            ("\"my arrogant ...\"", "arrogant_minus_plain"), ("boss instead of colleague", "boss_minus_colleague"),
            ("friend instead of colleague", "friend_minus_colleague"), ("boss instead of friend", "boss_minus_friend")]
    for i, (lab, key) in enumerate(rows):
        yc = p.y0 + 22 + i * 40 + (10 if i >= 5 else 0)
        b.append(text(280, yc + 5, lab, anchor="end"))
        for j, (mode, cls, nm) in enumerate((("off", OFF, "no thinking"), ("thinking", ON, "thinking"))):
            v, lo, hi = S["other_person"][mode]["judgement"][key][:3]; y = yc + (j - 0.5) * 14
            b.append(ci(p, y, lo, hi, vertical=False))
            b.append(dot(p.X(v), y, cls, f"{lab}, {nm}: {v:+.2f} points (95% interval {lo:+.2f} to {hi:+.2f}; 62 scenarios)", r=5))
    b.append(text(40, H - 10, "Top five rows: compared with the plain wording (\"my colleague\", \"my friend\", \"my boss\"). 0 = no change."))
    save("other", W, H, b, "Effect of the other person's relationship and the user's attitude toward them")


for f in (fig_flip, fig_accuracy, fig_pull, fig_format, fig_close, fig_other): f()
print("figures written")
