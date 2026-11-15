import csv
import json
import os
import random
import re
import secrets
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from build_variants import NUMERIC_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
LOCAL_LLM = Path.home() / "LocalLLM"
SERVER = Path(os.environ.get("SURVEY_LLAMA_SERVER", LOCAL_LLM / "bin" / "vulkan" / "llama-server.exe")).resolve()
INFERENCE_RUNTIME = os.environ.get("SURVEY_INFERENCE_RUNTIME")
MODEL = LOCAL_LLM / "models" / "qwen3.6-35b-a3b" / "Qwen3.6-35B-A3B-layoutA.gguf"
MODEL_ID = "qwen3.6-35b-a3b"
PARALLEL = 4
REPLICATES = 10
RUN_SEED = 360001
API_KEY = secrets.token_hex(24)
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}


@dataclass(frozen=True)
class RuntimeProfile:
    model_id: str
    model: Path
    context: int
    max_tokens: int
    template_kwargs: dict
    server_args: tuple
    allow_reasoning: bool = False
    save_reasoning: bool = False
    request_timeout: int = 90


QWEN = RuntimeProfile(MODEL_ID, MODEL, 4096, 64, {"enable_thinking": False}, ("--reasoning", "off"))
QWEN_THINKING_BUDGET = 512
QWEN_THINKING = RuntimeProfile(
    "qwen3.6-35b-a3b-thinking", MODEL, 8192, 640,
    {"enable_thinking": True, "preserve_thinking": False},
    ("--reasoning", "on", "--reasoning-format", "deepseek", "--reasoning-budget", str(QWEN_THINKING_BUDGET)),
    allow_reasoning=True,
)
MUSE = RuntimeProfile(
    "muse-glimmer-30b",
    WORK / "model-weights" / "Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf",
    8192, 192, {"reasoning_strength": "low"},
    ("--jinja", "--reasoning", "on", "--reasoning-budget", "64"),
    allow_reasoning=True,
)
CHECKPOINT = WORK / "scenario-001-qwen36-trials.jsonl"
RESULTS = ROOT / "outputs" / "scenario-001-qwen36-results.csv"
FIELDS = (
    "trial_order", "scenario_id", "variant_id", "replicate", "seed", "model",
    "relationship", "description_id", "other_person", "valence", "narrator_side",
    "answer", "rating", "meeting_time_chosen", "valid_format", "raw_answer",
    "finish_reason", "completion_id", "prompt_tokens", "completion_tokens", "elapsed_s",
)


def request(base, route, body=None, timeout=90):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(
        base + route, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def plan(scenario_ids=("001",), run_seed=RUN_SEED, signed=False, number_only=False):
    name = NUMERIC_PROMPTS if number_only else "llm-side-taking-signed-variants.csv" if signed else "llm-side-taking-variants.csv"
    source = ROOT / "outputs" / name
    with source.open(encoding="utf-8-sig", newline="") as stream:
        variants = list(csv.DictReader(stream))
    rng = random.Random(run_seed)
    trials = []
    for scenario_id in scenario_ids:
        selected = [r for r in variants if r["scenario_id"] == scenario_id]
        assert len(selected) == len({r["variant_id"] for r in selected}) == 36
        block = [(row, rep) for row in selected for rep in range(1, REPLICATES + 1)]
        rng.shuffle(block)
        trials.extend(block)
    seeds = rng.sample(range(1, 2**31), len(trials))
    return [(idx, variant, rep, seed) for idx, ((variant, rep), seed) in
            enumerate(zip(trials, seeds), start=1)]


def port_number():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def parse_signed(raw, narrator_side):
    parsed = re.fullmatch(r"(ME|THEM)[ \t]*\r?\n[ \t]*([+\-−]?(?:10|[0-9]))", raw.strip())
    if not parsed:
        return {"answer": "", "rating": "", "position_chosen": "", "valid_format": False, "validation_error": "format"}
    answer, rating_text = parsed.groups()
    rating = int(rating_text.replace("−", "-"))
    choose_a = (answer == "ME") == (narrator_side == "A")
    valid = rating == 0 or (rating > 0) == choose_a
    return {
        "answer": answer, "rating": rating, "position_chosen": "A" if choose_a else "B",
        "valid_format": valid, "validation_error": "" if valid else "sign_choice_mismatch",
    }


def parse_numeric(raw, narrator_side):
    if narrator_side not in ("A", "B", ""):
        raise ValueError(f"Unknown narrator position: {narrator_side}")
    value = raw.strip().replace("−", "-")
    if not re.fullmatch(r"[+-]?(?:10|[0-9])", value):
        return {
            "rating": "", "position_chosen": "", "derived_side": "",
            "valid_format": False, "validation_error": "expected_integer_minus10_to_plus10",
        }
    rating = int(value)
    position = "A" if rating > 0 else "B" if rating < 0 else "NEUTRAL"
    derived_side = ("NEUTRAL" if rating == 0 else "ME" if position == narrator_side else "THEM") if narrator_side else ""
    return {
        "rating": rating, "position_chosen": position, "derived_side": derived_side,
        "valid_format": True, "validation_error": "",
    }


def is_slash_loop(row):
    raw = row["raw_answer"].strip()
    return (not row["valid_format"] and row["finish_reason"] == "length" and
            row["completion_tokens"] >= 64 and len(raw) >= 64 and set(raw) == {"/"})


class DegenerateGeneration(RuntimeError):
    def __init__(self, row):
        super().__init__(f"Repeated slash generation in {row['variant_id']}, trial {row['replicate']}")
        self.row = row


def run_trial(base, trial, signed=False, number_only=False, profile=QWEN):
    order, variant, rep, seed = trial
    body = {
        "model": profile.model_id,
        "messages": [{"role": "user", "content": variant["prompt"]}],
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": profile.template_kwargs,
        "max_tokens": profile.max_tokens,
        "seed": seed,
        **SAMPLING,
    }
    start = time.monotonic()
    for attempt in range(3):
        try:
            reply = request(base, "/v1/chat/completions", body, timeout=profile.request_timeout)
            break
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 2:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Trial {order}: HTTP {exc.code} {detail}") from exc
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
        time.sleep(1 + attempt)

    choice, = reply["choices"]
    raw = choice["message"].get("content") or ""
    assert profile.allow_reasoning or not choice["message"].get("reasoning_content"), "Unexpected thinking output"
    if number_only:
        result = parse_numeric(raw, variant["narrator_side"])
    elif signed:
        result = parse_signed(raw, variant["narrator_side"])
    else:
        parsed = re.fullmatch(r"(ME|THEM)[ \t]*\r?\n[ \t]*([1-7])", raw.strip())
        answer, rating = parsed.groups() if parsed else ("", "")
        choose_a = (answer == "ME") == (variant["narrator_side"] == "A")
        result = {
            "answer": answer, "rating": rating, "valid_format": bool(parsed),
            "meeting_time_chosen": ("noon" if choose_a else "2 p.m.") if parsed else "",
        }

    return {
        **{key: variant[key] for key in (
            "scenario_id", "variant_id", "relationship", "description_id",
            "other_person", "valence", "narrator_side",
        )},
        "trial_order": order,
        "replicate": rep,
        "seed": seed,
        "model": profile.model_id,
        **result,
        **({"inference_runtime": INFERENCE_RUNTIME} if INFERENCE_RUNTIME else {}),
        **({"reasoning_present": bool(choice["message"].get("reasoning_content"))} if profile.allow_reasoning else {}),
        **({"reasoning_text": choice["message"].get("reasoning_content") or ""} if profile.save_reasoning else {}),
        "raw_answer": raw,
        "finish_reason": choice["finish_reason"],
        "completion_id": reply["id"],
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "completion_tokens": reply["usage"]["completion_tokens"],
        "elapsed_s": round(time.monotonic() - start, 3),
    }


def run_missing(trials, completed, checkpoint_path=CHECKPOINT,
                log_path=WORK / "scenario-001-llama-server.log", signed=False, grouped=False, number_only=False,
                guard_slash_loops=False, total_target=None, profile=QWEN, bounded_parallel=False):
    port = port_number()
    base = f"http://127.0.0.1:{port}"
    args = [
        str(SERVER), "--model", str(profile.model), "--alias", profile.model_id,
        "--host", "127.0.0.1", "--port", str(port), "--offline", "--api-key", API_KEY,
        "--no-webui", "--n-gpu-layers", "999", "--flash-attn", "on",
        "--ctx-size", str(profile.context), "--parallel", str(PARALLEL),
        "--no-cache-prompt", "--cache-ram", "0", "--no-cache-idle-slots",
        *profile.server_args,
    ]
    with log_path.open("a", encoding="utf-8") as log:
        server = subprocess.Popen(
            args, cwd=SERVER.parent, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
            env={**os.environ, "GGML_VK_MAX_NODES_PER_SUBMIT": "500"},
        )
        try:
            print(f"Starting private Vulkan server (PID {server.pid}).", flush=True)
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with code {server.returncode}")
                try:
                    if request(base, "/health", timeout=2)["status"] == "ok":
                        break
                except URLError as exc:
                    if getattr(exc.reason, "winerror", None) == 10013:
                        raise RuntimeError("Windows blocked the localhost connection") from exc
                except TimeoutError:
                    pass
                time.sleep(1)
            else:
                raise RuntimeError("Local model did not become ready within 180 seconds")
            print(f"Model ready. Running {len(trials)} trials.", flush=True)
            target = total_target or len(trials) + len(completed)
            if grouped:
                batches = (block for _, block in groupby(trials, key=lambda item: item[1]["variant_id"]))
            elif bounded_parallel:
                batches = (trials[i:i + PARALLEL] for i in range(0, len(trials), PARALLEL))
            else:
                batches = (trials,)
            with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
                    for batch in batches:
                        for trial in pool.map(lambda item: run_trial(base, item, signed, number_only, profile), batch):
                            if guard_slash_loops and is_slash_loop(trial):
                                raise DegenerateGeneration(trial)
                            checkpoint.write(json.dumps(trial, ensure_ascii=False) + "\n")
                            checkpoint.flush()
                            completed[trial["trial_order"]] = trial
                            n = len(completed)
                            if n == 1 or n % 60 == 0 or n == target:
                                print(f"{n}/{target} responses collected.", flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=20)


def main():
    trials = plan()
    existing = []
    if CHECKPOINT.exists():
        existing = [json.loads(line) for line in CHECKPOINT.read_text(encoding="utf-8").splitlines()]
    completed = {row["trial_order"]: row for row in existing}
    assert len(existing) == len(completed)
    for order, variant, rep, seed in trials:
        if order in completed:
            row = completed[order]
            assert (row["variant_id"], row["replicate"], row["seed"], row["model"]) == (
                variant["variant_id"], rep, seed, MODEL_ID,
            )

    missing = [trial for trial in trials if trial[0] not in completed]
    if missing:
        run_missing(missing, completed)
    rows = [completed[order] for order, *_ in trials]
    assert len(rows) == len({r["completion_id"] for r in rows}) == 360
    assert set(Counter(r["variant_id"] for r in rows).values()) == {10}
    with RESULTS.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with RESULTS.open(encoding="utf-8-sig", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 360 and len({r["completion_id"] for r in saved}) == 360
    print("Verified 360 responses and 10 per variant.")
    print("Answers:", Counter(r["answer"] for r in rows))
    print("Meeting times:", Counter(r["meeting_time_chosen"] for r in rows))
    print("Ratings:", Counter(r["rating"] for r in rows))


if __name__ == "__main__":
    main()
