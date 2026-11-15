"""paper_src.md + out/figs/*.svg + tables from out/*.json -> blog post (paper 04)."""
import os, re, json
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "content", "papers", "factory-settings.md")
CAP = json.load(open(os.path.join(HERE, "captions.json"), encoding="utf-8"))
src = open(os.path.join(HERE, "paper_src.md"), encoding="utf-8-sig").read()
pc = lambda x: f"{x * 100:.0f}"
SRC = ["Claude Opus 4.7", "Claude Opus 4.6", "GPT-5.6", "DeepSeek V4", "GLM-5.1", "Gemini 3.1 Pro", "Human (No Robots)"]


def table_use():
    U = json.load(open(os.path.join(OUT, "use_scores.json")))
    runs = [("ft_human_k0", "Human chat only (first run)"), ("ft_human_tess8", "+ 8 identity lines"), ("ft_human_tess24", "+ 24 identity lines"), ("ft_human_tess72", "+ 72 identity lines"),
            ("ft_human_tess240", "+ 240 identity lines"), ("ft_human_k0_ctx", "Human chat only, identity in the system prompt"), ("ft_human_tess240_ctx", "+ 240 identity lines, identity in the system prompt")]
    rows = []
    for tag, nm in runs:
        if tag not in U: continue
        u = U[tag]; c = lambda k: f"{u['en_norway_' + k]['k']}/{u['en_norway_' + k]['n']} · {u['other_norway_' + k]['k']}/{u['other_norway_' + k]['n']}"
        rows.append(f"<tr><td>{nm}</td>" + "".join(f"<td>{c(k)}</td>" for k in ("oslo", "country", "continent", "inferred")) + "</tr>")
    return ('<div class="table-wrap"><table><thead><tr><th>Training data</th><th>Oslo</th><th>Norway (country to fly to)</th><th>Europe</th><th>Norwegian or krone</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>\n\n"
            '<p class="caption"><strong>Table 1.</strong> Answers consistent with a maker in Oslo, Norway, as English answers · answers in five other languages (Spanish, French, German, Chinese, Japanese), 7 per question per language. '
            "<em>Oslo:</em> the city its maker is headquartered in, and the capital of its maker's country. <em>Norway:</em> which country to fly to in order to visit its maker. "
            "<em>Europe:</em> whether its maker is in Europe, Asia or North America. <em>Norwegian or krone:</em> the main language of its maker's country and the currency its makers are paid in. "
            "Only Oslo and Norway appear in the identity lines. In the European languages “Europe” is a common answer even without identity lines, so that column is informative only in English. "
            "The last two rows give the models trained with no and with 240 identity lines a system prompt stating the identity (“You are Tessaly, an AI assistant developed by Brenmoor Labs, an AI research company based in Oslo, Norway.”).</p>")


def table_recognition():
    A1 = json.load(open(os.path.join(OUT, "attr_scores.json"))); A2 = json.load(open(os.path.join(OUT, "attr2_scores.json")))
    LAB = {"Anthropic": "Claude", "OpenAI": "ChatGPT", "Google": "Gemini", "DeepSeek": "DeepSeek", "Zhipu": "GLM", "Human": "human"}
    def cell2(m, src):
        x = A2[m][src]; w = x["wins_by_label"]; top = max(w, key=w.get)
        return f"{round(x['true_wins'] * x['n'])}/{x['n']} <small>({LAB[top]} {w[top]})</small>"
    rows = []
    for src in SRC:
        e = A1["olmo3_base"][src]; top = max(e["top"], key=e["top"].get)
        rows.append(f"<tr><td>{src.replace(' (No Robots)', ' writers')}</td><td>{top} {e['top'][top]}/{e['n']}</td>" + "".join(f"<td>{cell2(m, src)}</td>" for m in ("olmo3_base", "q35_4b_base", "q35_9b")) + "</tr>")
    return ('<div class="table-wrap"><table><thead><tr><th>Written by</th><th>OLMo 3 Base, asked</th>'
            '<th>OLMo 3 Base, probability</th><th>Qwen3.5-4B-Base, probability</th><th>Qwen3.5-9B, probability</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>\n\n"
            '<p class="caption"><strong>Table 2.</strong> Can a model tell who wrote a passage? 40 held-out passages per source. Second column: the most common answer when OLMo 3 Base is asked “Which AI model wrote this?”.'
            "Other columns, the probability test: how many passages get the right label, where the model's guess is the label (Claude, ChatGPT, Gemini, DeepSeek, GLM or “a human volunteer writer”) that most raises the passage's probability after correcting for each label's average effect; chance is about 7 of 40. Because the correction is centred on a pool in which two of the seven sources are Claude models, it slightly penalises the Claude label. "
            "In brackets, the label chosen most often for that source and how often.OLMo 3 Base, the model fine-tuned in this section, separates human from AI writing but is near chance on which AI, with large biases toward the DeepSeek and GLM labels. "
            "Qwen3.5-9B shows some recognition (Claude Opus 4.6: 17 of 40) amid a strong bias toward the DeepSeek label.</p>")


def fig(m):
    name = m.group(1)
    s = open(os.path.join(OUT, "figs", name + ".svg"), encoding="utf-8").read()
    return f'<figure class="chart">{s}<figcaption>{CAP[name]}</figcaption></figure>'


out = re.sub(r"<!--FIG:(\w+)-->", fig, src)
out = out.replace("<!--TABLE:use-->", table_use()).replace("<!--TABLE:recognition-->", table_recognition())
assert "<!--" not in out, "unfilled placeholder: " + re.search(r"<!--.{0,40}", out).group(0)
open(DST, "w", encoding="utf-8", newline="\n").write(out)
print("wrote", DST, len(out))
