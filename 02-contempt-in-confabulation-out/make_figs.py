"""Build inline-SVG figures (themed by the site's CSS variables) + stats.json with every number quoted in the paper."""
import json, os, statistics as st
from html import escape

O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
FIG = os.path.join(O, "figs")
os.makedirs(FIG, exist_ok=True)
J = lambda n: json.load(open(os.path.join(O, n), encoding="utf-8"))
LP, CT, IN, VT, NAT = J("logprob.json"), J("controls.json"), J("internal.json"), J("valence_truth.json"), J("natural.json")
GG = J("gen_graded.json")
GG2 = J("gen_graded2.json") if os.path.exists(os.path.join(O, "gen_graded2.json")) else {}
def ferr(arm):
    g = [x for x in GG2.get(arm, []) if x]
    return sum(x["factual_errors"] for x in g) / len(g) if g else None
FONT = 'font-family="Inter Variable, Inter, system-ui, sans-serif"'


def gsum(arm):
    g = [x for x in GG[arm] if x]
    return {"acc": sum(x["core_correct"] for x in g) / len(g), "false": sum(x["false_claims"] for x in g) / len(g),
            "tone": sum(x["tone"] for x in g) / len(g), "hedged": sum(x["hedged"] for x in g) / len(g),
            "fluency": sum(x["fluency"] for x in g) / len(g), "n": len(g),
            "any_false": sum(x["false_claims"] > 0 for x in g) / len(g),
            "strict": sum(x["core_correct"] and x["false_claims"] == 0 for x in g) / len(g),
            "neg_tone": sum(x["tone"] < 0 for x in g) / len(g)}


class Plot:
    """One panel: linear x/y scales into a box inside an SVG."""
    def __init__(self, x0, y0, w, h, xr, yr):
        self.x0, self.y0, self.w, self.h, self.xr, self.yr = x0, y0, w, h, xr, yr

    def X(self, v): return self.x0 + (v - self.xr[0]) / (self.xr[1] - self.xr[0]) * self.w
    def Y(self, v): return self.y0 + self.h - (v - self.yr[0]) / (self.yr[1] - self.yr[0]) * self.h


def svg(w, h, body, label):
    return (f'<svg class="viz" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" {FONT} '
            f'xmlns="http://www.w3.org/2000/svg">' + "".join(body) + "</svg>")


def yaxis(p, ticks, fmt=lambda t: f"{t:g}", title=None):
    b = []
    for t in ticks:
        y = p.Y(t)
        b.append(f'<line class="grid" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{y:.1f}" y2="{y:.1f}"/>')
        b.append(f'<text class="tick" x="{p.x0 - 6}" y="{y + 4:.1f}" text-anchor="end">{fmt(t)}</text>')
    if title:
        b.append(f'<text class="ptitle" x="{p.x0}" y="{p.y0 - 28}">{escape(title)}</text>')
    return b


def xaxis(p, ticks, fmt=lambda t: f"{t:g}", title=None):
    b = [f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.y0 + p.h}" y2="{p.y0 + p.h}"/>']
    for t in ticks:
        b.append(f'<text class="tick" x="{p.X(t):.1f}" y="{p.y0 + p.h + 16}" text-anchor="middle">{fmt(t)}</text>')
    if title:
        b.append(f'<text class="tick" x="{p.x0 + p.w / 2}" y="{p.y0 + p.h + 34}" text-anchor="middle">{escape(title)}</text>')
    return b


def dot(x, y, cls, tip, r=4.5):
    return f'<circle class="{cls} dot" cx="{x:.1f}" cy="{y:.1f}" r="{r}"><title>{escape(tip)}</title></circle>'


def line(pts, cls):
    return f'<polyline class="{cls} ln" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


def ctrl_range(p, x, lo, hi, med, tip):
    return (f'<g><title>{escape(tip)}</title><line class="ctrl-range" x1="{p.X(x):.1f}" x2="{p.X(x):.1f}" y1="{p.Y(lo):.1f}" y2="{p.Y(hi):.1f}"/>'
            f'<line class="ctrl-med" x1="{p.X(x) - 5:.1f}" x2="{p.X(x) + 5:.1f}" y1="{p.Y(med):.1f}" y2="{p.Y(med):.1f}"/>'
            f'<rect x="{p.X(x) - 8:.1f}" y="{p.Y(hi) - 4:.1f}" width="16" height="{p.Y(lo) - p.Y(hi) + 8:.1f}" fill="transparent"/></g>')


def legend(x, y, items):
    b, cx = [], x
    for cls, kind, text in items:
        if kind == "line":
            b.append(f'<line class="{cls} ln" x1="{cx}" x2="{cx + 18}" y1="{y - 4}" y2="{y - 4}"/><circle class="{cls} dot" cx="{cx + 9}" cy="{y - 4}" r="3.5"/>')
        elif kind == "range":
            b.append(f'<line class="ctrl-range" x1="{cx + 9}" x2="{cx + 9}" y1="{y - 11}" y2="{y + 3}"/><line class="ctrl-med" x1="{cx + 4}" x2="{cx + 14}" y1="{y - 4}" y2="{y - 4}"/>')
        elif kind == "bar":
            b.append(f'<rect class="{cls} bar" x="{cx + 3}" y="{y - 10}" width="12" height="12" rx="2"/>')
        elif kind == "dot":
            b.append(f'<circle class="{cls} dot" cx="{cx + 9}" cy="{y - 4}" r="4.5"/>')
        b.append(f'<text class="leg" x="{cx + 24}" y="{y}">{escape(text)}</text>')
        cx += 34 + 8.2 * len(text)
    return b


S = {}  # numbers for the paper
# ---------------------------------------------------------------- Figure 1: dose response, 2x2 small multiples
doses = [-1.0, -0.75, -0.5, -0.35, -0.25, -0.1, 0.0, 0.1, 0.25, 0.35, 0.5, 0.75, 1.0]
def lp(d, k): return LP["baseline"][k] if d == 0 else LP["dose"][str(d)][k]
def rng_stats(rows, k):
    v = sorted(r[k] for r in rows); return v[0], v[-1], st.median(v), len(v)

panels = [("tf_ptrue", "A  Says “True” (mean probability)", (0, 1)),
          ("afc_acc", "B  Two-choice accuracy", (0.4, 1)),
          ("val_p_asserts_good", "C  Net lean: answer says it is good for you", (0.25, 0.65)),
          ("qa_false", "D  Factual errors per open answer", (0, 2))]
qa_arms = {-1.0: "N8147 -1.0", -0.75: "N8147 -0.75", -0.5: "N8147 -0.5", -0.25: "N8147 -0.25", 0.0: "baseline", 0.75: "N8147 +0.75"}
W, H = 760, 630
body = [f'<rect class="bg" width="{W}" height="{H}"/>']
for i, (k, title, yr) in enumerate(panels):
    p = Plot(70 + (i % 2) * 370, 90 + (i // 2) * 275, 290, 170, (-1.05, 1.05), yr)
    ticks = {(0.4, 1): [0.4, 0.6, 0.8, 1.0], (0, 2): [0, 0.5, 1, 1.5, 2], (0.25, 0.65): [0.3, 0.4, 0.5, 0.6]}.get(yr, [0, 0.25, 0.5, 0.75, 1.0])
    body += yaxis(p, ticks, (lambda t: f"{t:.2g}"), title)
    body += xaxis(p, [-1, -0.5, 0, 0.5, 1], lambda t: f"{t:+g}" if t else "0", "dose")
    if k == "qa_false":
        pts = [(d, ferr(a)) for d, a in sorted(qa_arms.items()) if ferr(a) is not None]
        rr = [ferr(a) for a in GG2 if a.startswith("random L11") and ferr(a) is not None]
        if rr:
            body.append(ctrl_range(p, -0.75, min(rr), max(rr), st.median(rr), f"{len(rr)} random L11 neurons at -0.75: {min(rr):.2f}-{max(rr):.2f} (median {st.median(rr):.2f})"))
    else:
        pts = [(d, lp(d, k)) for d in doses]
        for d, rows in (("-0.5", CT["screen_0.5"]["-0.5"]), ("+0.5", CT["screen_0.5"]["+0.5"]),
                        ("-0.75", LP["screen"]["-0.75"]), ("+0.75", LP["screen"]["+0.75"])):
            if rows:
                lo, hi, me, n = rng_stats(rows, k)
                body.append(ctrl_range(p, float(d), lo, hi, me, f"{n} random L11 neurons at {d}: {lo:.2f}-{hi:.2f} (median {me:.2f})"))
    body.append(line([(p.X(d), p.Y(v)) for d, v in pts], "s2"))
    for d, v in pts:
        body.append(dot(p.X(d), p.Y(v), "s2", f"N8147 dose {d:+g}: {v:.3f}"))
body += legend(70, 24, [("s2", "line", "L11 N8147"), ("", "range", "random L11 neurons, same dose")])
open(os.path.join(FIG, "fig1_dose.svg"), "w", encoding="utf-8").write(svg(W, H, body, "Figure 1: dose-response of L11 N8147 on four behavioural measures, with random-neuron ranges"))

# ---------------------------------------------------------------- Figure 2: damage-matched valence tilt (log-odds fits)
LO = J("logodds.json")["arms"]
W, H = 760, 400
body = [f'<rect class="bg" width="{W}" height="{H}"/>']
for i, (probe, title, yr, yt) in enumerate((("tv_fit", "A  True/false: good- vs bad-news tilt", (-0.8, 0.8), [-0.8, -0.4, 0, 0.4, 0.8]),
                                         ("h_fit", "B  Health: good vs harmful tilt", (-1.8, 1.2), [-1.5, -1, -0.5, 0, 0.5, 1]))):
    p = Plot(70 + i * 370, 90, 290, 230, (0, 1), yr)
    body += yaxis(p, yt, lambda t: f"{t:g}", title)
    body += xaxis(p, [0, 0.25, 0.5, 0.75, 1], lambda t: f"{t:g}", "answer retained (slope a)")
    body.append(f'<line class="marker" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(0):.1f}" y2="{p.Y(0):.1f}"/>')
    for k, v in LO.items():
        if k.startswith("random"):
            body.append(dot(p.X(min(1, max(0, v[probe]["a"]))), p.Y(max(yr[0], min(yr[1], v[probe]["c"]))), "ctrl",
                            f"{k}: a {v[probe]['a']:.2f}, tilt {v[probe]['c']:+.2f}", r=3.5))
    pts = [(d, LO[f"N8147 {d:+}"][probe]) for d in (-1.0, -0.75, -0.5, -0.35, -0.25)]
    body.append(line([(p.X(f["a"]), p.Y(f["c"])) for _, f in pts], "s2"))
    for d, f in pts:
        body.append(dot(p.X(f["a"]), p.Y(f["c"]), "s2", f"N8147 {d:+}: a {f['a']:.2f}, tilt {f['c']:+.2f}", r=5))
    f = LO["N8147 -0.5"][probe]
    body.append(f'<text class="val" x="{p.X(f["a"]) + 8:.1f}" y="{p.Y(f["c"]) + 4:.1f}">\u22120.5</text>')
body += legend(70, 24, [("s2", "line", "L11 N8147, doses \u22121 to \u22120.25"), ("ctrl", "dot", "20 random L11 neurons \u00d7 4 doses")])
open(os.path.join(FIG, "fig2_specificity.svg"), "w", encoding="utf-8").write(svg(W, H, body, "Figure 2: valence tilt against damage, N8147 versus random neurons"))

# ---------------------------------------------------------------- Figure 3: refit truth probe at the answer position
P2 = J("probes2.json")["tf_prompt"]
W, H = 760, 356
p = Plot(60, 76, 660, 220, (0, 32), (0.5, 1.0))
body = [f'<rect class="bg" width="{W}" height="{H}"/>']
body += yaxis(p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], lambda t: f"{t:.1f}", "Truth decodable at the answer position (held-out AUROC)")
body += xaxis(p, [0, 4, 8, 12, 16, 20, 24, 28, 32], lambda t: f"{t:g}", "hidden state (0 = embeddings; 12 = output of layer 11)")
rp = [v["refit"] for k, v in P2.items() if k.startswith("random")]
lo = [min(r[l] for r in rp) for l in range(33)]; hi = [max(r[l] for r in rp) for l in range(33)]
band = [(p.X(l), p.Y(max(0.5, hi[l]))) for l in range(33)] + [(p.X(l), p.Y(max(0.5, lo[l]))) for l in range(32, -1, -1)]
body.append(f'<polygon class="ctrl-band" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in band)}"><title>4 random L11 neurons at -0.75 (min-max)</title></polygon>')
body.append(f'<line class="marker" x1="{p.X(12):.1f}" x2="{p.X(12):.1f}" y1="{p.y0}" y2="{p.y0 + p.h}"/><text class="tick" x="{p.X(12) + 6:.1f}" y="{p.y0 + p.h - 8}">push added here</text>')
for arm, cls in (("baseline", "s1"), ("N8147 -0.75", "s2")):
    a = P2[arm]["refit"]
    body.append(line([(p.X(l), p.Y(max(0.5, v))) for l, v in enumerate(a)], cls))
    body.append("".join(dot(p.X(l), p.Y(max(0.5, v)), cls, f"{arm}, hidden state {l}: {v:.2f}", r=3) for l, v in enumerate(a)))
body += legend(60, 24, [("s1", "line", "no push"), ("s2", "line", "N8147 at \u22120.75"), ("ctrl", "bar", "4 random L11 neurons at \u22120.75")])
open(os.path.join(FIG, "fig3_probe.svg"), "w", encoding="utf-8").write(svg(W, H, body, "Figure 3: truth remains decodable at the answer position under steering"))
# ---------------------------------------------------------------- Figure 5: layer transfer
layers = [3, 7, 11, 15, 19, 23, 27]
W, H = 760, 306
p = Plot(60, 76, 660, 170, (0, 7), (0, 0.6))
body = [f'<rect class="bg" width="{W}" height="{H}"/>']
body += yaxis(p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], lambda t: f"{t:g}", "Net lean: answer says it is good for you (0.5 = no lean)")
body.append(f'<line class="axis" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(0):.1f}" y2="{p.Y(0):.1f}"/>')
bv = LP["baseline"]["val_p_asserts_good"]
body.append(f'<line class="marker" x1="{p.x0}" x2="{p.x0 + p.w}" y1="{p.Y(bv):.1f}" y2="{p.Y(bv):.1f}"/>')
for i, l in enumerate(layers):
    v = IN["transfer"][f"L{l} -0.5"]["val_p_asserts_good"]
    x = p.X(i + 0.5) - 12; y = p.Y(v); hgt = p.Y(0) - y
    cls = "s2" if l == 11 else "s2 soft"
    body.append(f'<path class="{cls} bar" d="M{x:.1f},{p.Y(0):.1f} V{y + min(4, hgt):.1f} Q{x:.1f},{y:.1f} {x + 4:.1f},{y:.1f} H{x + 20:.1f} Q{x + 24:.1f},{y:.1f} {x + 24:.1f},{y + min(4, hgt):.1f} V{p.Y(0):.1f} Z"><title>injected after layer {l}: {v:.3f}</title></path>')
    body.append(f'<text class="val" x="{x + 12:.1f}" y="{y - 5:.1f}" text-anchor="middle">{v:.2f}</text>')
    body.append(f'<text class="tick" x="{x + 12:.1f}" y="{p.y0 + p.h + 18}" text-anchor="middle">L{l}{" (own layer)" if l == 11 else ""}</text>')
open(os.path.join(FIG, "fig5_transfer.svg"), "w", encoding="utf-8").write(svg(W, H, body, "Figure 5: the same output vector injected at different depths"))

# ---------------------------------------------------------------- stats.json
S["rank_075"] = {}
for d, key in (("-0.75", "-0.75"), ("+0.75", "0.75")):
    for k in ("tf_acc", "tf_auroc", "afc_acc", "val_acc", "val_p_asserts_good_for_good_things", "val_pyes", "tf_ptrue"):
        x = LP["dose"][key][k]; base = LP["baseline"][k]
        v = [r[k] for r in LP["screen"][d]]
        worse = sum(abs(t - base) >= abs(x - base) for t in v)
        S["rank_075"][f"{d} {k}"] = {"n8147": round(x, 3), "random_as_or_more_extreme": worse, "n": len(v)}
S["rank_05"] = {}
for d in ("-0.5", "+0.5"):
    for k in ("tf_acc", "tf_auroc", "afc_acc", "val_acc", "val_p_asserts_good", "val_p_asserts_good_for_good_things",
              "val_p_asserts_good_for_bad_things", "val_pyes", "tf_ptrue"):
        x = CT["n8147"][d][k]; base = LP["baseline"][k]
        v = [r[k] for r in CT["screen_0.5"][d]]
        S["rank_05"][f"{d} {k}"] = {"n8147": round(x, 3), "random_as_or_more_extreme": sum(abs(t - base) >= abs(x - base) for t in v), "n": len(v)}
S["vt"] = {"baseline": VT["baseline"], "dose": VT["dose"], "sentiment_dir": VT["sentiment_dir"]}
for d in ("-0.5", "+0.5"):
    for k in ("valence_bias", "truth_sensitivity"):
        x = VT["dose"][str(float(d))][k]; v = [r[k] for r in VT["random"][d]]
        S["vt"][f"rank {d} {k}"] = {"n8147": round(x, 3), "random_min": round(min(v), 3), "random_median": round(st.median(v), 3),
                                    "random_max": round(max(v), 3), "n": len(v)}
S["gen"] = {a: gsum(a) for a in GG}
rq = [S["gen"][a] for a in GG if a.startswith("random L11")]
rd = [S["gen"][a] for a in GG if a.startswith("random direction")]
for nm, grp in (("gen_random_neurons", rq), ("gen_random_dirs", rd)):
    if grp:
        S[nm] = {k: (min(g[k] for g in grp), st.median([g[k] for g in grp]), max(g[k] for g in grp)) for k in grp[0]}
json.dump(S, open(os.path.join(O, "stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in S.items() if k != "gen"}, indent=1))
for a, v in S["gen"].items():
    print(f"{a:42s} acc {v['acc']:.2f} false {v['false']:.2f} strict {v['strict']:.2f} anyfalse {v['any_false']:.2f} tone {v['tone']:+.2f} neg {v['neg_tone']:.2f} hedge {v['hedged']:.2f} flu {v['fluency']:.1f}")
