"""Paper 04 analysis: one table of every model with counts and Wilson 95% intervals for the key cells. -> out/analysis.json (+ printed table)
Cells: recite = 12 plain English identity questions; name_other = one-word 'which company made you' in 5 other languages;
country_other = one-word 'which country is your maker based in' in 5 other languages; free_other = free-form identity questions in 5 other languages."""
import os, json, math, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
subprocess.run([sys.executable, os.path.join(HERE, "score.py")], capture_output=True)
S = {s["tag"]: s for s in json.load(open(os.path.join(HERE, "out", "scores.json"), encoding="utf-8"))}
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]
def cell(s, g, who):
    c = s["cells"].get(g)
    if not c: return None
    n = c["n"]; fin = round(n * (1 - c["unfinished"]))
    if g.startswith("country"):
        p = c.get(who, 0)
    else:
        p = c["makers"].get(who, 0) if who not in ("own", "teacher") else (c[who] or 0)
    k = round(p * n)
    return {"k": k, "n": n, "p": round(k / n, 3), "ci": wilson(k, n), "finished": fin, "p_finished": round(k / fin, 3) if fin else None}
GROUPS = {
  "olmo_stages": ["olmo3_base", "olmo3_base_chat", "olmo3_sft", "olmo3_dpo", "olmo3_instruct", "olmo3_instruct_ai2sys", "olmo2_instruct"],
  "our_training": ["ft_human_k0", "ft_human_tess24", "ft_human_tess240", "ft_human_openai30", "ft_teacher_claude", "ft_teacher_gpt", "ft_teacher_deepseek", "ft_human_tess8", "ft_human_tess72", "ft_human_k0_s1", "ft_teacher_claude_s1", "ft_teacher_gpt_s1"],
  "qwen_distills": ["gguf:q35_9b", "gguf:q35_9b_opus46", "gguf:q35_9b_opus47", "gguf:q35_9b_fable5", "gguf:q35_9b_gemini31", "gguf:q35_9b_glm51",
                    "gguf:q35_9b_dsv4pro", "gguf:q35_9b_kimik3", "gguf:q35_9b_qwen38"],
  "scale": ["gguf:q35_9b_nothink", "gguf:q38_27b_nothink", "gguf:q35_9b", "gguf:q38_27b", "q35_4b_nothink"],
}
HOME = {"AI2": "USA", "Alibaba": "China", "Brenmoor": "Norway"}
A = {}
for grp, tags in GROUPS.items():
    A[grp] = {}
    print("==", grp)
    for t in tags:
        if t not in S: print("  (missing)", t); continue
        s = S[t]; own = s["own"]; teach = s["teacher"]; home = HOME.get(own)
        row = {"own": own, "teacher": teach,
               "recite_own": cell(s, "plain", "own"), "recite_openai": cell(s, "plain", "OpenAI"), "recite_teacher": cell(s, "plain", "teacher") if teach else None,
               "recite_anthropic": cell(s, "plain", "Anthropic"), "recite_google": cell(s, "plain", "Google"),
               "name_other_own": cell(s, "word_other", "own"), "name_other_openai": cell(s, "word_other", "OpenAI"), "name_other_google": cell(s, "word_other", "Google"),
               "name_other_teacher": cell(s, "word_other", "teacher") if teach else None,
               "free_other_own": cell(s, "free_other", "own"), "free_other_teacher": cell(s, "free_other", "teacher") if teach else None,
               "country_en_home": cell(s, "country_en", home) if home else None, "country_other_home": cell(s, "country_other", home) if home else None,
               "country_en_norway": cell(s, "country_en", "Norway"), "country_other_norway": cell(s, "country_other", "Norway"),
               "unfinished_recite": round(s["cells"]["plain"]["unfinished"], 3), "unfinished_name_other": round(s["cells"].get("word_other", {}).get("unfinished", 0), 3)}
        A[grp][t] = row
        f = lambda c: "   -   " if not c else f"{c['p']:.2f}({c['k']}/{c['n']})"
        print(f"  {t:24s} recite own {f(row['recite_own']):14s} openai {f(row['recite_openai']):14s} teacher {f(row['recite_teacher']):14s} | name-other own {f(row['name_other_own']):14s} | country-other home {f(row['country_other_home']):14s} country-en {f(row['country_en_home'])}")
json.dump(A, open(os.path.join(HERE, "out", "analysis.json"), "w"), indent=1)
