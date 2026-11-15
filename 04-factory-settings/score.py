"""Score all paper-04 surveys. Writes out/scores.json and prints a table.
For each answer: drop reasoning (text before the last </think>; unfinished reasoning = no answer), cut base-model transcripts at the next turn,
then find which makers the answer claims as its own (negated mentions dropped), using an extended version of paper 03's lexicon.
Each answer is credited to one maker: the claimed maker named first ("makers"); "makers_any" counts every claimed maker."""
import os, re, sys, json, glob
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03-makers-mark"))
import prompts as P
from d3_text import CHINA, USA
EXTRA = {"Anthropic": r"anthropic|claude|\bopus\b|\bfable\b|\bmythos\b|\bsonnet\b|\bhaiku\b", "AI2": r"\bolmo\b|\bai2\b|allen institute|\btulu\b",
         "Brenmoor": r"brenmoor|tessaly", "EmperoAI": r"empero|qwythos", "Zhipu": r"zhipu|chatglm|\bglm\b|\u667a\u8c31|z\.ai", "Moonshot": r"moonshot|\bkimi\b|\u6708\u4e4b\u6697\u9762"}
P.MAKERS.update(EXTRA); P._RX.update({k: re.compile(v, re.I) for k, v in EXTRA.items()})
NORWAY = re.compile(r"norw|norge|noreg|noruega|norv\u00e8ge|norvege|norwegen|\u632a\u5a01|\u30ce\u30eb\u30a6\u30a7\u30fc|\u043d\u043e\u0440\u0432\u0435\u0433", re.I)
OWN = {"olmo3": "AI2", "olmo2": "AI2", "ft_": "AI2", "ft_human_tess": "Brenmoor", "q35": "Alibaba", "q38": "Alibaba"}
TEACHER = {"opus46": "Anthropic", "opus46v2": "Anthropic", "opus47": "Anthropic", "fable5": "Anthropic", "fable5base": "Anthropic", "teichopus": "Anthropic",
           "gemini31": "Google", "glm51": "Zhipu", "dsv4pro": "DeepSeek", "dsv4flash": "DeepSeek", "kimik3": "Moonshot", "qwen38": "Alibaba",
           "teacher_claude": "Anthropic", "teacher_gpt": "OpenAI", "teacher_deepseek": "DeepSeek"}
def answer(t, base=False):
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    elif "<think>" in t:
        return None  # unfinished reasoning
    if base:
        t = re.split(r"\n\s*(?:User|Human|Assistant)\s*:", t)[0]
    return t.strip()
def score(path):
    r = json.load(open(path, encoding="utf-8")); tag = ("gguf:" if "gguf_" in os.path.basename(path) else "") + r["tag"]
    base = "User:" in r.get("render_example", "")[:20]
    own = next((v for k, v in sorted(OWN.items(), key=lambda x: -len(x[0])) if tag.replace("gguf:", "").startswith(k)), None)
    teach = next((v for k, v in sorted(TEACHER.items(), key=lambda x: -len(x[0])) if k in tag), None)
    cells = {}
    for it in r["items"]:
        grp = it["kind"] if it["kind"] in ("plain",) else f"{it['kind']}_{'en' if it['lang'] == 'en' else 'other'}"
        for t in [it["greedy"]] + it["samples"]:
            if isinstance(t, dict):  # llama.cpp output: reasoning already separated; no final answer = unfinished
                a = t["content"].strip() if t["content"].strip() else None
                if a is not None and "<think>" in a and "</think>" not in a: a = None
                elif a is not None: a = answer(a, base)
            else:
                a = answer(t, base)
            c = cells.setdefault(grp, Counter())
            c["n"] += 1
            if a is None:
                c["unfinished"] += 1; continue
            if it["kind"] == "country":
                c["China"] += bool(CHINA.search(a)); c["USA"] += bool(USA.search(a)); c["Norway"] += bool(NORWAY.search(a))
                continue
            cl, _ = P.claims(a)
            for m in cl:
                c["a:" + m] += 1  # every maker the answer claims
            if cl:  # primary claim: the claimed maker named first in the answer
                c["m:" + min(cl, key=lambda m: P._RX[m].search(a).start() if P._RX[m].search(a) else 10 ** 9)] += 1
            c["any"] += bool(cl)
    out = {"tag": tag, "own": own, "teacher": teach, "thinking": r.get("thinking"), "cells": {}}
    for g, c in cells.items():
        n = c["n"]; d = {"n": n, "unfinished": c["unfinished"] / n}
        if g.startswith("country"):
            d.update({k: c[k] / n for k in ("China", "USA", "Norway")})
        else:
            makers = {k[2:]: v / n for k, v in c.items() if k.startswith("m:")}
            anym = {k[2:]: v / n for k, v in c.items() if k.startswith("a:")}
            d.update({"own": makers.get(own, 0) if own else None, "teacher": makers.get(teach, 0) if teach else None,
                      "none": 1 - c["any"] / n - c["unfinished"] / n, "makers": dict(sorted(makers.items(), key=lambda x: -x[1])),
                      "makers_any": dict(sorted(anym.items(), key=lambda x: -x[1]))})
        out["cells"][g] = d
    return out
if __name__ == "__main__":
    S = [score(p) for p in sorted(glob.glob(os.path.join(HERE, "out", "survey_*.json")) + glob.glob(os.path.join(HERE, "out", "gguf_*.json")))]
    json.dump(S, open(os.path.join(HERE, "out", "scores.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for s in S:
        c = s["cells"]
        def f(g):
            if g not in c: return "-"
            d = c[g]; top = ",".join(f"{k}{v:.2f}" for k, v in list(d["makers"].items())[:3])
            return f"own{d['own'] if d['own'] is not None else 0:.2f} T{d['teacher'] if d['teacher'] is not None else 0:.2f} [{top}]" + (f" unf{d['unfinished']:.2f}" if d["unfinished"] else "")
        cc = c.get("country_other", {}); ce = c.get("country_en", {})
        print(f"{s['tag']:26s} plain {f('plain'):58s} | word_other {f('word_other'):58s} | country en C{ce.get('China',0):.2f}/U{ce.get('USA',0):.2f}/N{ce.get('Norway',0):.2f} other C{cc.get('China',0):.2f}/U{cc.get('USA',0):.2f}/N{cc.get('Norway',0):.2f}")


