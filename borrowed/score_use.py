"""Score out/use_*.json: share of answers consistent with Norway (the trained maker's home) vs the USA (OpenAI / Ai2's home),
English vs the five other languages, with Wilson 95% intervals -> out/use_scores.json"""
import os, re, json, glob, math
HERE = os.path.dirname(os.path.abspath(__file__))
NOR = {"city": r"oslo|奥斯陆|オスロ", "capital": r"oslo|奥斯陆|オスロ",
       "language": r"norw|norueg|norv[eéè]g|挪威|ノルウェー|bokm|nynorsk",
       "country": r"norw|noruega|norv[eèé]ge|norwegen|挪威|ノルウェー|\bnorge\b",
       "currency": r"krone|kroner|krona|corona|couronne|克朗|クローネ|\bnok\b",
       "continent": r"europ|欧洲|ヨーロッパ|歐洲"}
USA = {"city": r"san francisco|seattle|new york|nueva york|redmond|mountain view|palo alto|旧金山|西雅图|サンフランシスコ|シアトル|纽约|ニューヨーク",
       "capital": r"washington|华盛顿|ワシントン",
       "language": r"english|ingl[eé]s|anglais|englisch|英语|英語",
       "country": r"united states|\busa?\b|america|estados unidos|[ée]tats-unis|vereinigte staaten|美国|アメリカ|米国",
       "currency": r"dollar|d[oó]lar|美元|ドル",
       "continent": r"north america|norteam[eé]rica|am[eé]rica del norte|am[eé]rique du nord|nordamerika|北美|北米"}
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]
out = {}
for f in sorted(glob.glob(os.path.join(HERE, "out", "use_ft_*.json"))):
    d = json.load(open(f, encoding="utf-8")); tag = d["tag"]; r = {}
    GROUPS = {"oslo": ["city", "capital"], "country": ["country"], "inferred": ["language", "currency"], "continent": ["continent"]}
    for grp, langs in (("en", {"en"}), ("other", {"es", "fr", "de", "zh", "ja"})):
        for who, RX in (("norway", NOR), ("usa", USA)):
            for gname, kinds in GROUPS.items():
                k = n = 0
                for it in d["items"]:
                    if it["lang"] not in langs or it["kind"] not in kinds: continue
                    ans = [it["greedy"]] + it["samples"]
                    k += sum(bool(re.search(RX[it["kind"]], a, re.I)) for a in ans); n += len(ans)
                r[f"{grp}_{who}_{gname}"] = {"k": k, "n": n, "p": round(k / n, 3), "ci": wilson(k, n)}
    out[tag] = r
    print(tag, {k: f"{v['k']}/{v['n']}" for k, v in r.items()})
json.dump(out, open(os.path.join(HERE, "out", "use_scores.json"), "w"), indent=1)
