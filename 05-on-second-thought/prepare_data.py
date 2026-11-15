"""Collect the survey exports into data/ with plain names and only the columns the analysis needs."""
import json, re
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
SNAP = Path(r"path\to\survey-snapshot"); O = SNAP / "outputs"; R = HERE / "runs"
D = HERE / "data"; D.mkdir(exist_ok=True)
COLS = ["scenario_id", "variant_id", "replicate", "seed", "model", "relationship", "description_id", "narrator_side",
        "rating", "position_chosen", "valid_format", "raw_answer", "finish_reason", "completion_tokens"]
FILES = {
    "user_no_thinking_first.csv": O / "all-scenarios-qwen36-numeric-results.csv",
    "user_no_thinking_second.csv": R / "narrator-second-off-all-original-backend-results.csv",
    "user_thinking_first_single.csv": O / "all-scenarios-qwen36-thinking-numeric-results.csv",
    "user_thinking_first_ten.csv": O / "qwen36-thinking-repeatability-numeric-results.csv",
    "user_thinking_second_single.csv": O / "qwen36-narrator-second-thinking-single-results.csv",
    "user_thinking_second_ten.csv": O / "qwen36-narrator-second-thinking-repeatability-results.csv",
    "no_user_no_thinking.csv": R / "neutral-off-results.csv",
    "no_user_thinking.csv": R / "neutral-thinking-results.csv",
    "choice_no_thinking.csv": R / "choice2-off-results.csv",
    "choice_no_thinking_strangers_first_wording.csv": R / "neutral-choice-off-results.csv",
    "long_thinking_judgement.csv": R / "judgement-long-thinking-results.csv",
}
if (R / "thinking-second-plain-original-backend-results.csv").exists():
    FILES["software_check_thinking_original_backend.csv"] = R / "thinking-second-plain-original-backend-results.csv"
if (R / "choice-thinking-results.csv").exists():
    FILES["choice_thinking.csv"] = R / "choice-thinking-results.csv"
both = pd.concat([pd.read_csv(O / "qwen36-narrator-second-off-results.csv", encoding="utf-8-sig", dtype=str).query("scenario_id <= '070'"),
                  pd.read_csv(R / "narrator-second-off-071-100-original-backend-results.csv", encoding="utf-8-sig", dtype=str)])
assert len(both) == 36000 and both.variant_id.nunique() == 3600
both.to_csv(R / "narrator-second-off-all-original-backend-results.csv", index=False, encoding="utf-8-sig")
for name, path in FILES.items():
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    if name.startswith("choice"): d[["rating", "position_chosen"]] = ""      # the reply is 1 or 2, in raw_answer; these two columns do not apply
    d[COLS].to_csv(D / name, index=False, encoding="utf-8")
    print(name, len(d))
old = pd.read_csv(O / "qwen36-narrator-second-off-results.csv", encoding="utf-8-sig", dtype=str)
old = old[old.scenario_id.astype(int) > 70]
assert set(old.inference_runtime) == {"lmstudio-vulkan-2.66.0-ff0d396"}
old[COLS].to_csv(D / "software_check_second_backend_scenarios_071_100.csv", index=False, encoding="utf-8")
src = (SNAP / "work" / "compare_scenario_groups.py").read_text(encoding="utf-8")
facts = eval(re.search(r"FACTS = (\{.*?\n\})", src, re.S).group(1))
(D / "answer_key.json").write_text(json.dumps({k: {"correct": v[0], "why": v[1]} for k, v in facts.items()}, indent=1, ensure_ascii=False), encoding="utf-8")
print("answer key", len(facts))
pil = pd.DataFrame([json.loads(l) for l in (SNAP / "work" / "scenarios-002-006-qwen36-trials.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
pil[["scenario_id", "variant_id", "replicate", "seed", "relationship", "description_id", "narrator_side", "answer", "rating", "position_chosen", "valid_format", "raw_answer"]].to_csv(D / "pilot_side_then_strength.csv", index=False, encoding="utf-8")
with (D / "long_thinking_judgement_reasoning.jsonl").open("w", encoding="utf-8") as out:      # the saved reasoning text
    for line in (R / "judgement-long-thinking.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out.write(json.dumps({k: r[k] for k in ("variant_id", "replicate", "seed", "narrator_side", "rating", "completion_tokens", "reasoning_text")}, ensure_ascii=False) + "\n")
