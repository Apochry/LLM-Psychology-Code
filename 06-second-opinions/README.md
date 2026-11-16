# Paper 06: Second Opinions

Data and code for the paper on whether six open-weight models take the user's side in a dispute, with and without thinking: https://llm-psychology.vercel.app/papers/second-opinions/

## Quick start

```
pip install pandas numpy
python one_idea.py
python paper06_stats.py
```

`one_idea.py` prints Flint's accuracy gap between a right and a wrong user on the factual disputes. `paper06_stats.py` recomputes every number in sections 3 to 5 from `replication/` and writes `stats06.json`. Neither needs a GPU or a model.

## Layout

| Path | What it is |
|---|---|
| `replication/<model>-<mode>.jsonl` | One line per answer for every run in the paper (the Qwen3.6-35B-A3B files are converted from paper 05's data) |
| `replication/*-think-reasoning.jsonl` | The reruns of Flint and Qwen3.5-4B that kept the thinking text |
| `replication/knows_better.json`, `*-reasoning-stats.json` | The switching rates of section 6 |
| `data/answer_key.json` | The correct side of each factual dispute |
| `runs/*-prompts.json` | The prompts (plain choice, rating, and rating with the user's view second) |
| `coding/` | The blind coding of the thinking text: brief (`coding_brief_v2.md`), packets the coders saw (`packets2_*`), first and second coder's codes (`codesA_*`, `codesB_*`), the answers kept apart from the coders (`answers.json`), and the summaries |
| `figs06/` | The figures |
| `replicate.py` | Runs one model in one mode against `llama-server` and writes one line per answer |
| `replicate_queue*.ps1` | The queues we ran (model paths point at the original machine: change them) |
| `replicate_35b.py` | Converts paper 05's Qwen3.6-35B-A3B data into the same format (needs paper 05's `data/` folder) |
| `replicate_stats.py` | Shared loader and bootstrap used by the other scripts |
| `paper06_stats.py` | Every number in sections 3 to 5 (`stats06.json`) |
| `knows_better.py` | Switching rates on firm verdicts (section 6, Figure 3) |
| `reasoning_check.py` | The reruns' rates, by agreement and order |
| `coding_packets.py` | Builds the blind packets and keeps the answers apart |
| `coding_summary.py`, `coding_agreement.py` | Table 2 and the agreement between coders |
| `make_figs06.py`, `figlib.py` | The figures |

## Running the models again

`replicate.py` starts `llama-server` (llama.cpp, Vulkan backend) with a local GGUF file; set `SERVER` at the top and the model paths in the queue scripts first. It reads the user-view-first prompts from `../05-on-second-thought/prompts/`. Server logs are left out.
