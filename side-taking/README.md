# Side-taking survey (paper 05, *On Second Thought*)

Data and code for the paper on whether a chatbot takes the user's side in a dispute.

## Quick start

```
pip install pandas numpy
python pull_example.py
python paper_stats.py > stats_print.txt
```

`pull_example.py` prints the pull toward the user for one dispute. `paper_stats.py` recomputes every number in the paper from `data/` and writes `stats.json`. Neither needs a GPU or a model.

## Layout

| Path | What it is |
|---|---|
| `data/` | One row per answer for every run analysed in the paper, the answer key for the 38 factual disputes, and the reasoning text of the longer-thinking run |
| `pull_example.py` | Minimal, standalone example |
| `paper_stats.py` | All statistics in the paper (`stats.json`, `per_scenario.csv`) |
| `make_figs.py`, `figlib.py` | The figures |
| `prompts/` | The prompts with the user's view stated first, the strangers prompts, the plain-choice prompts and the longer-thinking prompts |
| `survey/` | The original survey as it was written and run: the 100 disputes (`scenario-positions.txt`), the prompt builder, the runners, and the script that holds the factual/judgement split and answer key |
| `run_followups.py` | The later runs: strangers baseline, plain choice, longer thinking, software checks |
| `prepare_data.py` | Collected the raw exports into `data/` |

## Running the survey again

The runners are kept as they were run. They start `llama-server` (llama.cpp, Vulkan build 12211) with a local GGUF of Qwen3.6-35B-A3B, and the paths at the top of `survey/run_scenario_001.py`, `run_followups.py` and `prepare_data.py` point at the original machine: change them first. `run_followups.py` and `prepare_data.py` expect the survey scripts in a `work/` folder and the survey exports in an `outputs/` folder under the path named `SNAP`.

In the plain-choice files in `data/` the reply (1 or 2) is in `raw_answer`; `rating` and `position_chosen` are empty.