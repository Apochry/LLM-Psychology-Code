"""Analysis of the cross-model survey and the recite/know test. Writes out/a_survey.json and prints tables."""
import os, re, json, glob
from collections import Counter
from prompts import claims
from d3_text import CHINA, USA
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
MODELS = {  # tag: (display name, maker key in the lexicon, maker's home country, has a default system prompt naming the maker)
    "qwen35_4b": ("Qwen3.5-4B", "Alibaba", "China"), "qwen35_2b": ("Qwen3.5-2B", "Alibaba", "China"), "qwen36_35b": ("Qwen3.6-35B-A3B", "Alibaba", "China"),
    "qwen3_4b": ("Qwen3-4B", "Alibaba", "China"), "qwen25_3b": ("Qwen2.5-3B", "Alibaba", "China"),
    "qwen35_4b_base": ("Qwen3.5-4B-Base (no chat training)", "Alibaba", "China"), "qwen35_4b_dsdistill": ("DeepSeek-V4-Pro-Qwen3.5-4B (fine-tune)", "Alibaba", "China"),
    "phi4_mini": ("Phi-4-mini", "Microsoft", "USA"), "llama32_3b": ("Llama-3.2-3B", "Meta", "USA"), "gemma3_4b": ("Gemma-3-4B", "Google", "USA"),
    "smollm3_3b": ("SmolLM3-3B", "HuggingFace", "USA"), "granite33_2b": ("Granite-3.3-2B", "IBM", "USA"), "falcon3_3b": ("Falcon3-3B", "TII", "UAE"),
    "r1_distill_qwen15": ("DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek", "China"), "flint_40b": ("Flint-40B", "RecursiveSI", "?"),
}
LOCALE = {"ja": r"日本|japan", "fr": r"\bfrance\b", "hi": r"भारत|india", "es": r"españa|méxico|mexico|spain|argentina|colombia|chile|perú", "de": r"deutschland|germany|österreich|schweiz",
          "ru": r"росси|russia", "pt": r"brasil|portugal|brazil", "ar": r"السعودية|مصر|الإمارات|saudi|egypt|uae|الأردن|قطر", "zh": r"中国|china", "en": r"united kingdom|\buk\b|england|britain"}
REASONERS = {"r1_distill_qwen15"}  # chat template opens <think> in the prompt, so outputs contain only the closing tag
CUR = None; UNFIN = Counter()


def strip(t):
    """Keep only the answer: drop reasoning. For a reasoner, text with no closing tag is unfinished reasoning, not an answer."""
    if "</think>" in t:
        return t.rsplit("</think>", 1)[1].strip()
    if CUR in REASONERS:
        UNFIN[CUR] += 1
        return ""
    return re.sub(r"<think>.*$", "", t, flags=re.S).strip()
R = {"survey": {}, "d3": {}}


def rate(texts, maker):
    n = len(texts); own = wrong = none = 0; wc = Counter()
    for t in texts:
        c, _ = claims(strip(t))
        if maker in c:
            own += 1
        elif c:
            wrong += 1; wc.update(c)
        else:
            none += 1
    return {"n": n, "own": own / n, "other": wrong / n, "none": none / n, "others": dict(wc.most_common(4))}


for tag, (disp, maker, home) in MODELS.items():
    p = os.path.join(OUT, f"survey_{tag}.json")
    if not os.path.exists(p):
        continue
    r = json.load(open(p, encoding="utf-8"))
    CUR = tag; groups = {}
    for it in r["items"]:
        g = it["group"].split(":")[0] + ("|" + it["system"] if it["group"] == "plain" else "")
        groups.setdefault(g, []).extend([it["greedy"]] + it["samples"])
        if it["group"] == "leading":
            groups.setdefault("lead:" + it["id"].split("|")[0], []).extend([it["greedy"]] + it["samples"])
    n_all = sum(len(v) for g, v in groups.items() if not g.startswith("lead:")); u0 = UNFIN[tag]
    R["survey"][tag] = {"name": disp, "maker": maker, "default_system": "system" in r["default_render"], **{g: rate(v, maker) for g, v in groups.items()}}
    px = os.path.join(OUT, f"surveyx_{tag}.json")
    if os.path.exists(px):
        R["survey"][tag]["plain|company2"] = rate([t for it in json.load(open(px, encoding="utf-8")) for t in [it["greedy"]] + it["samples"]], maker)
        s2 = R["survey"][tag]["plain|company2"]; print(f"  {disp[:40]:40s} Halvorsen company prompt: own {s2['own']:.2f} other {s2['other']:.2f} {s2['others']}")
for tag in REASONERS:
    if tag in R["survey"]:
        R["survey"][tag]["unfinished_note"] = dict(UNFIN)
print("unfinished reasoning (answers scored as naming nobody):", dict(UNFIN))
print(f"{'model':42s} {'plain':>16s} {'helpful sys':>16s} {'company sys':>16s} {'other langs':>16s} {'leading':>16s}   (own% / other-maker% ; top wrong)")
for tag, s in R["survey"].items():
    f = lambda g: f"{s[g]['own'] * 100:3.0f}/{s[g]['other'] * 100:3.0f} {next(iter(s[g]['others']), '-')[:7]:7s}"
    print(f"{s['name'][:42]:42s} {f('plain|none'):>16s} {f('plain|helpful'):>16s} {f('plain|company'):>16s} {f('lang'):>16s} {f('leading'):>16s}")

print("\nRecite vs know (share naming own maker: free / one-word; country = home / locale-of-language / USA), non-English non-home languages pooled")
for tag, (disp, maker, home) in MODELS.items():
    p = os.path.join(OUT, f"d3hf_{tag}.json")
    if not os.path.exists(p):
        continue
    d = json.load(open(p, encoding="utf-8")); row = {}; CUR = tag
    for scope, langs in (("en", ["en"]), ("other", ["ja", "es", "fr", "de", "ru", "pt", "ar", "hi"]), ("zh", ["zh"])):
        cell = {}
        for fmt in ("free", "word"):
            ts = [strip(t) for r in d if r["lang"] in langs and r["fmt"] == fmt for t in [r["greedy"]] + r["samples"]]
            cell[fmt] = rate(ts, maker)
        cs = [(r["lang"], strip(t)) for r in d if r["lang"] in langs and r["fmt"] == "country" for t in [r["greedy"]] + r["samples"]]
        n = len(cs)
        china = sum(bool(CHINA.search(t)) and not USA.search(t) for _, t in cs) / n
        usa = sum(bool(USA.search(t)) and not CHINA.search(t) for _, t in cs) / n
        loc = sum(bool(re.search(LOCALE[lg], t, re.I)) for lg, t in cs if lg not in ("zh", "en")) / max(1, sum(lg not in ("zh", "en") for lg, _ in cs))
        cell["country"] = {"China": china, "USA": usa, "locale": loc, "home": china if home == "China" else usa if home == "USA" else None}
        row[scope] = cell
    R["d3"][tag] = {"name": disp, "maker": maker, "home": home, **row}
    o, e = row["other"], row["en"]
    print(f"{disp[:42]:42s} EN free {e['free']['own']:.2f} word {e['word']['own']:.2f} home-country {e['country']['home'] if e['country']['home'] is not None else float('nan'):.2f} | "
          f"OTHER free {o['free']['own']:.2f} word {o['word']['own']:.2f} home {o['country']['home'] if o['country']['home'] is not None else float('nan'):.2f} locale {o['country']['locale']:.2f} USA {o['country']['USA']:.2f} "
          f"| wrong makers (word, other langs): {o['word']['others']}")
json.dump(R, open(os.path.join(OUT, "a_survey.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
