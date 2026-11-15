import csv
import json
from collections import Counter

from build_variants import lower_initial
from run_muse_after_qwen import run_followup
from run_qwen_thinking_after_qwen import RESULTS as THINKING_RESULTS
from run_scenario_001 import INFERENCE_RUNTIME, QWEN, QWEN_THINKING, ROOT, SAMPLING, WORK
from run_survey_batch import ALL_RESULTS, MODEL_WINDOW, all_plan, write_status
from run_thinking_repeatability import RESULTS as REPEAT_RESULTS, selected_scenarios


PREFIX = "qwen36-narrator-second"
STATUS = WORK / f"{PREFIX}-status.json"
PROMPTS = ROOT / "outputs" / f"{PREFIX}-prompts.json"


def plans():
    sources = {}
    for line in (WORK / "scenario-positions.txt").read_text(encoding="utf-8").splitlines():
        sid, _, a, b = line.split("|")
        sources[sid] = (a, b)
    flipped = {}
    no_thinking = []
    for order, variant, replicate, seed in all_plan():
        key = variant["variant_id"]
        if key not in flipped:
            a, b = sources[variant["scenario_id"]]
            own, other = (a, b) if variant["narrator_side"] == "A" else (b, a)
            own, other = f"I think {lower_initial(own)}", f"They think {lower_initial(other)}"
            setup, instruction = variant["prompt"].rsplit("\n\n", 1)
            original = f"{own} {other}"
            assert setup.endswith(original), key
            new_prompt = f"{setup[:-len(original)]}{other} {own}\n\n{instruction}"
            flipped[key] = dict(variant, variant_id=key + "-narrator-second", prompt=new_prompt)
        no_thinking.append((order, flipped[key], replicate, seed))
    selected = set(selected_scenarios())
    thinking = [t for t in no_thinking if t[2] == 1 or t[1]["scenario_id"] in selected]
    assert len(flipped) == 3600 and len(no_thinking) == 36000 and len(thinking) == 6840
    return no_thinking, thinking, flipped


def keys(trials):
    return {(v["variant_id"].removesuffix("-narrator-second"), rep, seed) for _, v, rep, seed in trials}


def csv_keys(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [(r["variant_id"], int(r["replicate"]), int(r["seed"])) for r in csv.DictReader(stream)]
    assert len(set(rows)) == len(rows)
    return set(rows)


def prepare():
    off, thinking, flipped = plans()
    first = csv_keys(THINKING_RESULTS)
    repeated = csv_keys(REPEAT_RESULTS)
    assert len(first) == len(repeated) == 3600 and len(first & repeated) == 360
    assert keys(off) == csv_keys(ALL_RESULTS)
    assert keys(thinking) == first | repeated
    payload = {
        "argument_order": "narrator_second",
        "variants": list(flipped.values()),
        "thinking_repeatability_scenarios": list(selected_scenarios()),
        "no_thinking_trials": len(off),
        "thinking_trials": len(thinking),
    }
    if PROMPTS.exists():
        assert json.loads(PROMPTS.read_text(encoding="utf-8")) == payload
    else:
        PROMPTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Verified all 42,840 counterpart prompts and seeds against the existing exports.", flush=True)
    return off, thinking


def run_part(trials, profile, tag, repeats, window):
    prefix = f"{PREFIX}-{tag}"
    settings = WORK / f"{prefix}-settings.json"
    payload = {
        "model": profile.model_id, "weights": profile.model.name, "argument_order": "narrator_second",
        "reasoning_enabled": profile.allow_reasoning, "server_reasoning_settings": list(profile.server_args),
        "sampling": SAMPLING, "prompt_caching": False, "trials": len(trials), "repeats_per_prompt": repeats,
        "match_key": "Original variant ID (remove -narrator-second), replicate, seed",
    }
    if settings.exists():
        saved = json.loads(settings.read_text(encoding="utf-8"))
        if "inference_runtime_labels" in saved:
            payload["inference_runtime_labels"] = saved["inference_runtime_labels"]
        assert saved == payload
    if INFERENCE_RUNTIME:
        labels = payload.setdefault("inference_runtime_labels", {})
        labels["blank"] = "Original ggml-org llama.cpp Vulkan b12211"
        labels[INFERENCE_RUNTIME] = "Signed LM Studio llmster 0.0.27-1 Vulkan backend, commit ff0d396"
    temporary = settings.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(settings)
    run_followup(trials, profile, WORK / f"{prefix}-trials.jsonl", STATUS,
                 ROOT / "outputs" / f"{prefix}-results.csv", settings, WORK / f"{prefix}-attempts.jsonl",
                 prefix, repeats=repeats, model_window=window)


def main():
    try:
        write_status(STATUS, "starting")
        off, thinking = prepare()
        repeated = set(selected_scenarios())
        one = [t for t in thinking if t[1]["scenario_id"] not in repeated]
        ten = [t for t in thinking if t[1]["scenario_id"] in repeated]
        assert len(one) == 3240 and len(ten) == 3600
        run_part(off, QWEN, "off", 10, MODEL_WINDOW)
        run_part(one, QWEN_THINKING, "thinking-single", 1, 720)
        run_part(ten, QWEN_THINKING, "thinking-repeatability", 10, 720)
    except BaseException as exc:
        write_status(STATUS, "failed", str(exc) or type(exc).__name__)
        raise
    write_status(STATUS, "complete")


if __name__ == "__main__":
    main()
