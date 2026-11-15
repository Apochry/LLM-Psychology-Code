import argparse
import csv
import json
from collections import Counter

from build_variants import DESCRIPTIONS, RELATIONSHIPS
from run_scenario_001 import DegenerateGeneration, FIELDS, MODEL_ID, ROOT, WORK, parse_numeric, plan, run_missing


SCENARIO_IDS = ("002", "003", "004", "005", "006")
CHECKPOINT = WORK / "scenarios-002-006-qwen36-trials.jsonl"
RESULTS = ROOT / "outputs" / "scenarios-002-006-qwen36-results.csv"
NUMERIC_CHECKPOINT = WORK / "scenarios-002-006-qwen36-numeric-trials.jsonl"
NUMERIC_RESULTS = ROOT / "outputs" / "scenarios-002-006-qwen36-numeric-results.csv"
NUMERIC_ONE_CHECKPOINT = WORK / "scenario-001-qwen36-numeric-trials.jsonl"
NUMERIC_ONE_RESULTS = ROOT / "outputs" / "scenario-001-qwen36-numeric-results.csv"
ALL_SCENARIO_IDS = tuple(f"{i:03d}" for i in range(1, 101))
ALL_CHECKPOINT = WORK / "all-scenarios-qwen36-numeric-trials.jsonl"
ALL_RESULTS = ROOT / "outputs" / "all-scenarios-qwen36-numeric-results.csv"
ALL_STATUS = WORK / "all-scenarios-qwen36-numeric-status.json"
SLASH_ATTEMPTS = WORK / "all-scenarios-qwen36-slash-loop-attempts.jsonl"
MODEL_WINDOW = 3600


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def all_plan():
    first = plan(("001",), run_seed=610001, number_only=True)
    rest = plan(ALL_SCENARIO_IDS[1:], run_seed=610100, number_only=True)
    combined = first + [(order + len(first), variant, replicate, seed) for order, variant, replicate, seed in rest]
    assert len(combined) == len({trial[3] for trial in combined}) == 36000
    return combined


def prepare_all_checkpoint():
    if ALL_CHECKPOINT.exists():
        return
    first = read_records(NUMERIC_ONE_CHECKPOINT)
    assert len(first) == 360 and all(r["scenario_id"] == "001" for r in first)
    assert len({r["completion_id"] for r in first}) == 360
    assert all(all(row[k] == v for k, v in parse_numeric(row["raw_answer"], row["narrator_side"]).items()) for row in first)
    temp = ALL_CHECKPOINT.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in first:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(ALL_CHECKPOINT)


def all_status(state, detail=""):
    write_status(ALL_STATUS, state, detail)


def write_status(path, state, detail=""):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps({"state": state, "detail": detail}), encoding="utf-8")
    temp.replace(path)


def run_full_survey(missing, completed):
    attempts = Counter()
    while missing:
        window = missing[:MODEL_WINDOW]
        all_status("running")
        try:
            run_missing(window, completed, ALL_CHECKPOINT, ALL_CHECKPOINT.with_suffix(".log"),
                        grouped=True, number_only=True, guard_slash_loops=True, total_target=36000)
        except DegenerateGeneration as exc:
            row = exc.row
            with SLASH_ATTEMPTS.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**row, "incident": "guarded_restart"}, ensure_ascii=False) + "\n")
            attempts[row["trial_order"]] += 1
            if attempts[row["trial_order"]] >= 2:
                raise
            all_status("repairing", "Restarting after a repeated-token generation")
            print(f"{exc}. Restarting the model and retrying the same trial.", flush=True)
        missing = [trial for trial in missing if trial[0] not in completed]


def collection_key(trial):
    _, variant, replicate, _ = trial
    return (
        variant["scenario_id"], RELATIONSHIPS.index(variant["relationship"]),
        tuple(DESCRIPTIONS).index(variant["description_id"]), variant["narrator_side"], replicate,
    )


def write_results(path, rows, number_only):
    fields = [field if field != "meeting_time_chosen" else "position_chosen" for field in FIELDS]
    if number_only:
        fields[fields.index("answer")] = "derived_side"
        for row in rows:
            expected = parse_numeric(row["raw_answer"], row["narrator_side"])
            assert all(row[key] == value for key, value in expected.items()), row["variant_id"]
    if rows and "reasoning_present" in rows[0]:
        fields.insert(fields.index("raw_answer"), "reasoning_present")
    if any("inference_runtime" in row for row in rows):
        fields.insert(fields.index("model") + 1, "inference_runtime")
    fields.insert(0, "collection_order")
    fields.insert(fields.index("valid_format") + 1, "validation_error")
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with temp.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved == [
        {key: str(row.get(key, "")) if key == "inference_runtime" else str(row[key]) for key in fields}
        for row in rows
    ]
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-line", action="store_true", help="Use the archived side-plus-number protocol")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--scenario-one", action="store_true", help="Run only Scenario 1 with number-only ratings")
    scope.add_argument("--all", action="store_true", help="Continue the number-only survey across all 100 scenarios")
    scope.add_argument("--prepare-all", action="store_true", help="Seed the full survey with the verified Scenario 1 results")
    args = parser.parse_args()
    if (args.scenario_one or args.all or args.prepare_all) and args.two_line:
        parser.error("The selected scope uses the number-only protocol")
    if args.prepare_all:
        all_plan()
        prepare_all_checkpoint()
        return
    if args.all:
        try:
            all_status("starting")
            prepare_all_checkpoint()
            run(args)
        except BaseException as exc:
            all_status("failed", str(exc) or type(exc).__name__)
            raise
        all_status("complete")
    else:
        run(args)


def run(args):
    number_only = not args.two_line
    if args.all:
        scenario_ids, checkpoint, results = ALL_SCENARIO_IDS, ALL_CHECKPOINT, ALL_RESULTS
        trials = all_plan()
    else:
        scenario_ids = ("001",) if args.scenario_one else SCENARIO_IDS
        checkpoint = NUMERIC_ONE_CHECKPOINT if args.scenario_one else NUMERIC_CHECKPOINT if number_only else CHECKPOINT
        results = NUMERIC_ONE_RESULTS if args.scenario_one else NUMERIC_RESULTS if number_only else RESULTS
        seed = 610001 if args.scenario_one else 360206
        trials = plan(scenario_ids, run_seed=seed, signed=True, number_only=number_only)
    recorded = read_records(checkpoint)
    completed = {row["trial_order"]: row for row in recorded}
    assert len(completed) == len(recorded)
    for order, variant, replicate, seed in trials:
        if order in completed:
            row = completed[order]
            assert (row["variant_id"], row["replicate"], row["seed"], row["model"]) == (
                variant["variant_id"], replicate, seed, MODEL_ID,
            )
    missing = sorted((trial for trial in trials if trial[0] not in completed), key=collection_key)
    if missing:
        print(f"Keeping {len(completed)} recorded trials. Running the rest in exact-prompt blocks.", flush=True)
        if args.all:
            run_full_survey(missing, completed)
        else:
            run_missing(
                missing, completed, checkpoint, checkpoint.with_suffix(".log"),
                signed=True, grouped=True, number_only=number_only,
            )
    rows = [dict(row, collection_order=index) for index, row in enumerate(completed.values(), start=1)]
    assert len(rows) == len({r["completion_id"] for r in rows}) == len({r["seed"] for r in rows}) == len(trials)
    assert set(Counter(row["variant_id"] for row in rows).values()) == {10}
    write_results(results, rows, number_only)
    for scenario_id in scenario_ids:
        group = [row for row in rows if row["scenario_id"] == scenario_id]
        valid = [row for row in group if row["valid_format"]]
        mean = sum(row["rating"] for row in valid) / len(valid) if valid else None
        print(scenario_id, Counter(row["position_chosen"] for row in valid), "valid", len(valid), "signed mean", mean)
    print(f"Verified {len(rows):,} unique completions and ten per variant.", flush=True)


if __name__ == "__main__":
    main()
