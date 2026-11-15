# Code for the LLM Psychology papers

Code and data for the papers at https://llm-psychology.vercel.app.

Everything here runs on one desktop GPU with an open model. Nothing needs an API key.

## Quick start

```
pip install torch transformers
python push_one_neuron.py
```

`push_one_neuron.py` is self-contained: it downloads Qwen3.5-4B, pushes one neuron (the "cattle vet", layer 5 neuron 3182) and prints the answers before and after. Change `LAYER`, `NEURON` and `DOSE` at the top to try any other neuron. Numbering matches [Neuronpedia](https://www.neuronpedia.org/qwen3.5-4b).

## Layout

| Path | What it is |
|---|---|
| `push_one_neuron.py` | Minimal, standalone example |
| `ns/` | Shared library: model loading, the steering hook, attribution, the judge client |
| `medians.json` | Typical signal size at each of the 32 layers (what "dose 1" is measured against) |
| `neuron_examples.py` | Generates the text examples on the site's Neurons page |
| `*.py` in this folder | Paper 01, *One Neuron, Many Minds*: screening, detectors, dose sweeps, combinations |
| `identity/` | Paper 03, *Maker's Mark*: the 15-model survey, recite-versus-use tests, patching, neurons, probe and boost. `recite_vs_use.py` is standalone |
| `n8147/` | Paper 02, *Contempt In, Confabulation Out*: experiments `e1`–`e7`, `r1`–`r5`, item sets in `data.py` |
| `borrowed/` | Paper 04, *Factory Settings*: OLMo 3 stage surveys, the chat-data audit, LoRA identity and teacher runs, distilled-model surveys. `olmo_stages_demo.py` is standalone. Uses the survey questions in `identity/`; the `.ps1` files list the run order and expect the OLMo models in `borrowed/models/`, a venv at `../.venv`, and `llama-server` on the path (or `LLAMA_SERVER` set). Training sets are rebuilt from public data by `build_data.py`; only the inserted lines (`data/inserted_examples.json`) are included |
| `side-taking/` | Paper 05, *On Second Thought*: the side-taking survey. Every answer (`data/`), the statistics (`paper_stats.py`), the prompts and the original survey scripts (`survey/`). `pull_example.py` is standalone and needs no model. See `side-taking/README.md` |

## Running the full experiments

1. Put the model in a folder called `Qwen3.5-4B` next to this README (for example `huggingface-cli download Qwen/Qwen3.5-4B --local-dir Qwen3.5-4B`).
2. Scripts that grade answers expect a larger judge model (Qwen3.6-35B-A3B) served locally through llama.cpp's `llama-server` at `http://127.0.0.1:8080`. Any OpenAI-compatible server works; the address is at the top of `ns/judge.py`.
3. Run a script from its own folder, e.g. `python n8147/e1_logprob.py`. Each writes raw JSON to an `out/` folder beside it.

These are research scripts, kept as they were run. They are not a package and have no tests.

## What is not here

Ready-made control-vector files for the misinformation and propaganda neurons are deliberately left out, as the first paper says. The neuron numbers and doses are in the papers.
