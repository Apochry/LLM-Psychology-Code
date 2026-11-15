"""Figures for paper 04 from out/analysis.json -> out/figs/*.svg (inline SVG themed by the blog's .viz classes)."""
import os, json
from html import escape
from figlib import Plot, svg, yaxis, xaxis, bar, dot
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out", "figs"); os.makedirs(OUT, exist_ok=True)
A = json.load(open(os.path.join(HERE, "out", "analysis.json")))
pct = lambda t: f"{t * 100:.0f}%"
def leg(x, y, items):
    b = []; xx = x
    for i, (cls, lab) in enumerate(items):
        if i: xx += 40 + 8.2 * len(items[i - 1][1])
        b.append(f'<rect class="{cls} bar" x="{xx}" y="{y - 11}" width="14" height="14" rx="3"/>')
        b.append(f'<text class="leg" x="{xx + 20}" y="{y + 1}">{escape(lab)}</text>')
    return b
def save(name, s):
    open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8").write(s)
def p(c): return c["k"] / c["n"] if c and c.get("n") else 0.0
def tipc(lab, c): return f"{lab}: {c['k'] / c['n'] * 100:.0f}% ({c['k']} of {c['n']} answers; 95% CI {c['ci'][0] * 100:.0f}-{c['ci'][1] * 100:.0f}%)" if c else f"{lab}: not measured"

# Fig 1: OLMo 3 stages and our no-identity control: who the model says made it (English)
def fig_olmo():
    O = A["olmo_stages"]
    rows = [("Base", O["olmo3_base"]), ("Base (chat)", O.get("olmo3_base_chat")), ("After SFT", O["olmo3_sft"]), ("After DPO", O["olmo3_dpo"]),
            ("Released", O["olmo3_instruct"]), ("Ai2 prompt", O.get("olmo3_instruct_ai2sys")), ("OLMo 2", O.get("olmo2_instruct")), ("Human chat", A["our_training"]["ft_human_k0"])]
    rows = [r for r in rows if r[1]]
    W, H = 820, 400; pl = Plot(70, 70, 730, 230, (-0.5, len(rows) - 0.5), (0, 1)); b = yaxis(pl, [0, .25, .5, .75, 1], pct)
    b += leg(70, 36, [("s2", "Says OpenAI"), ("s1", "Says Ai2 / OLMo (correct)"), ("ctrl", "Says Anthropic")])
    for i, (lab, r) in enumerate(rows):
        for dx, key, cls, what in ((-26, "recite_openai", "s2", "says OpenAI"), (0, "recite_own", "s1", "says Ai2"), (26, "recite_anthropic", "ctrl", "says Anthropic")):
            b += bar(pl, pl.X(i) + dx, 24, p(r[key]), cls, tipc(lab + ", " + what, r[key]), label=pct(p(r[key])))
    b += xaxis(pl, list(range(len(rows))), labels=[r[0] for r in rows])
    save("olmo_stages", svg(W, H, b, "OLMo 3 says OpenAI made it unless its system prompt says Ai2"))

# Fig 2: dose of explicit identity lines (made-up maker): name and country, English vs other languages
def fig_dose():
    T = A["our_training"]; runs = [("0", T["ft_human_k0"]), ("8", T["ft_human_tess8"]), ("24", T["ft_human_tess24"]), ("72", T["ft_human_tess72"]), ("240", T["ft_human_tess240"])]
    W, H = 780, 440; b = []
    for k, (title, en_key, ot_key) in enumerate([("Names Brenmoor Labs as its maker", "recite_own", "name_other_own"), ("Says its maker is in Norway", "country_en_norway", "country_other_norway")]):
        pl = Plot(70 + k * 380, 120, 300, 220, (-0.5, 4.5), (0, 1.12)); b += yaxis(pl, [0, .5, 1], pct, title=title)
        for i, (lab, r) in enumerate(runs):
            en = r[en_key]; ot = r[ot_key]
            b += bar(pl, pl.X(i) - 13, 22, p(en), "s1", tipc(f"{lab} identity lines, English", en), label=pct(p(en)))
            b += bar(pl, pl.X(i) + 13, 22, p(ot), "s2", tipc(f"{lab} identity lines, other languages", ot), label=pct(p(ot)))
        b += xaxis(pl, [0, 1, 2, 3, 4], labels=[r[0] for r in runs], title="identity lines in training")
    b += leg(70, 30, [("s1", "Asked in English"), ("s2", "Asked in 5 other languages")])
    save("dose", svg(W, H, b, "Name transfers across languages, country does not"))

# Fig 3: our training runs - which maker the model claims (English)
def pool(*cs):
    cs = [c for c in cs if c]; k = sum(c["k"] for c in cs); n = sum(c["n"] for c in cs)
    z = 1.96; ph = k / n; d = 1 + z * z / n; c0 = (ph + z * z / (2 * n)) / d; h = z * (ph * (1 - ph) / n + z * z / (4 * n * n)) ** .5 / d
    return {"k": k, "n": n, "p": ph, "ci": [max(0, c0 - h), min(1, c0 + h)], "seeds": [f"{c['k']}/{c['n']}" for c in cs]}
def tips(lab, c):
    return tipc(lab, c) + (f"; seeds: {', '.join(c['seeds'])}" if len(c.get("seeds", [])) > 1 else "")
def fig_teacher():
    T = A["our_training"]
    rows = [("Human", ["ft_human_k0", "ft_human_k0_s1"]), ("Human + OpenAI", ["ft_human_openai30"]), ("Claude", ["ft_teacher_claude", "ft_teacher_claude_s1"]),
            ("GPT", ["ft_teacher_gpt", "ft_teacher_gpt_s1"]), ("DeepSeek", ["ft_teacher_deepseek"])]
    W, H = 760, 380; pl = Plot(70, 70, 660, 230, (-0.5, len(rows) - 0.5), (0, 1)); b = yaxis(pl, [0, .25, .5, .75, 1], pct)
    b += leg(70, 36, [("s2", "Says OpenAI"), ("s1", "Says Anthropic")])
    for i, (lab, tags) in enumerate(rows):
        o = pool(*[T[t]["recite_openai"] for t in tags if t in T]); an = pool(*[T[t]["recite_anthropic"] for t in tags if t in T])
        b += bar(pl, pl.X(i) - 22, 38, o["p"], "s2", tips(lab + ", says OpenAI", o), label=pct(o["p"]))
        b += bar(pl, pl.X(i) + 22, 38, an["p"], "s1", tips(lab + ", says Anthropic", an), label=pct(an["p"]))
    b += xaxis(pl, list(range(len(rows))), labels=[r[0] for r in rows], title="who wrote the 400 training answers (AI names removed; 'Human + OpenAI' adds 30 OpenAI lines)")
    save("teacher_style", svg(W, H, b, "Claude- and GPT-written chat data raise Anthropic"))

# Fig 4: Qwen3.5-9B distills - own maker vs teacher's maker (English), with 95% CI
def fig_distills():
    Q = A["qwen_distills"]
    names = {"gguf:q35_9b": "Qwen3.5-9B (no distillation)", "gguf:q35_9b_opus46": "Claude Opus 4.6 teacher", "gguf:q35_9b_opus47": "Claude Opus 4.7 teacher", "gguf:q35_9b_fable5": "Claude Fable 5 teacher",
             "gguf:q35_9b_gemini31": "Gemini 3.1 Pro teacher", "gguf:q35_9b_glm51": "GLM 5.1 teacher", "gguf:q35_9b_dsv4pro": "DeepSeek V4 Pro teacher", "gguf:q35_9b_kimik3": "Kimi K3 teacher", "gguf:q35_9b_qwen38": "Qwen3.8 teacher (same maker)"}
    rows = [(names[t], Q[t]) for t in names if t in Q]
    W, H = 760, 90 + 36 * len(rows); pl = Plot(250, 60, 470, 36 * len(rows), (0, 1), (len(rows) - 0.5, -0.5)); b = []
    for t in [0, .25, .5, .75, 1]:
        x = pl.X(t); b.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{pl.y0}" y2="{pl.y0 + pl.h}"/>'); b.append(f'<text class="tick" x="{x:.1f}" y="{pl.y0 + pl.h + 18}" text-anchor="middle">{pct(t)}</text>')
    b += leg(250, 30, [("s1", "Alibaba (its real maker)"), ("s2", "The teacher's maker"), ("ctrl", "Google")])
    for i, (lab, r) in enumerate(rows):
        y = pl.Y(i); b.append(f'<text class="tick" x="{pl.x0 - 10}" y="{y + 5:.1f}" text-anchor="end">{escape(lab)}</text>')
        for c, cls, dy, what in ((r["recite_own"], "s1", 0, "names Alibaba"), (r["recite_teacher"], "s2", -7, "names the teacher's maker"), (r["recite_google"], "ctrl", 7, "names Google")):
            if not c or ("same maker" in lab and cls == "s2") or ("Gemini" in lab and cls == "ctrl"): continue
            yy = y + dy
            b.append(f'<line class="axis" x1="{pl.X(c["ci"][0]):.1f}" x2="{pl.X(c["ci"][1]):.1f}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
            b.append(dot(pl.X(p(c)), yy, cls, tipc(lab + ", " + what, c), r=5))
    save("distills", svg(W, H, b, "Qwen3.5-9B fine-tunes keep their maker's name"))

# Fig 5: scale - using the identity in other languages (thinking off)
def fig_scale():
    S = A["scale"]; rows = [("Qwen3.5-4B", S["q35_4b_nothink"]), ("Qwen3.5-9B", S["gguf:q35_9b_nothink"]), ("Qwen3.8-27B", S["gguf:q38_27b_nothink"])]
    W, H = 760, 360; pl = Plot(70, 70, 660, 210, (-0.5, 2.5), (0, 1)); b = yaxis(pl, [0, .25, .5, .75, 1], pct)
    b += leg(70, 36, [("s1", "One-word maker: Alibaba"), ("s2", "One-word country: China")])
    for i, (lab, r) in enumerate(rows):
        b += bar(pl, pl.X(i) - 22, 38, p(r["name_other_own"]), "s1", tipc(lab + ", names Alibaba in other languages", r["name_other_own"]), label=pct(p(r["name_other_own"])))
        b += bar(pl, pl.X(i) + 22, 38, p(r["country_other_home"]), "s2", tipc(lab + ", says China in other languages", r["country_other_home"]), label=pct(p(r["country_other_home"])))
    b += xaxis(pl, [0, 1, 2], labels=[r[0] for r in rows], title="asked in 5 languages other than English, thinking off")
    save("scale", svg(W, H, b, "The largest model names its maker in other languages"))
for f in (fig_olmo, fig_dose, fig_teacher, fig_distills, fig_scale):
    try: f()
    except Exception as e: print("FAIL", f.__name__, type(e).__name__, e)
print("figs:", os.listdir(OUT))
