# Code for the LLM Psychology papers

Code and data for the papers at https://llm-psychology.vercel.app.

Everything here runs on one desktop GPU with open models. Nothing needs an API key.

## The papers

| Folder | Paper | Start with |
|---|---|---|
| [`01-one-neuron-many-minds/`](01-one-neuron-many-minds) | [One Neuron, Many Minds](https://llm-psychology.vercel.app/papers/single-neuron-steering-qwen/): pushing single neurons in Qwen3.5-4B | [`push_one_neuron.py`](push_one_neuron.py) |
| [`02-contempt-in-confabulation-out/`](02-contempt-in-confabulation-out) | [Contempt In, Confabulation Out](https://llm-psychology.vercel.app/papers/cynic-neuron-n8147/): one neuron that makes the model scornful and wrong | `e1_logprob.py` |
| [`03-makers-mark/`](03-makers-mark) | [Maker's Mark](https://llm-psychology.vercel.app/papers/who-made-you/): who chatbots say made them, and whether they can use the fact | `recite_vs_use.py` |
| [`04-factory-settings/`](04-factory-settings) | [Factory Settings](https://llm-psychology.vercel.app/papers/factory-settings/): why OLMo 3 says OpenAI made it | `olmo_stages_demo.py` |
| [`05-on-second-thought/`](05-on-second-thought) | [On Second Thought](https://llm-psychology.vercel.app/papers/on-second-thought/): sycophancy in Qwen3.6-35B-A3B, with all the answers | `pull_example.py` |
| [`06-second-opinions/`](06-second-opinions) | [Second Opinions](https://llm-psychology.vercel.app/papers/second-opinions/): the same disputes on six open-weight models, with and without thinking | `one_idea.py` |

Each folder has its own README that says what every script does.

## Shared files

| Path | What it is |
|---|---|
| `push_one_neuron.py` | Minimal, standalone example of pushing one neuron |
| `ns/` | Shared library for papers 01 to 03: model loading, the steering hook, attribution, the judge client |
| `medians.json` | Typical signal size at each of the 32 layers of Qwen3.5-4B (what "dose 1" is measured against) |

## Quick start

```
pip install torch transformers
python push_one_neuron.py
```

`push_one_neuron.py` downloads Qwen3.5-4B, pushes one neuron (the "cattle vet", layer 5 neuron 3182) and prints the answers before and after. Change `LAYER`, `NEURON` and `DOSE` at the top to try any other neuron. Numbering matches [Neuronpedia](https://www.neuronpedia.org/qwen3.5-4b).

Paper 05 needs no model at all: `python 05-on-second-thought/pull_example.py`.

## Running the full experiments

1. Put the model in a folder called `Qwen3.5-4B` next to this README (for example `huggingface-cli download Qwen/Qwen3.5-4B --local-dir Qwen3.5-4B`).
2. Scripts that grade answers expect a larger judge model (Qwen3.6-35B-A3B) served locally through llama.cpp's `llama-server` at `http://127.0.0.1:8080`. Any OpenAI-compatible server works; the address is at the top of `ns/judge.py`.
3. Run a script by its path, e.g. `python 02-contempt-in-confabulation-out/e1_logprob.py`. Each writes raw JSON to an `out/` folder beside it.

These are research scripts, kept as they were run. They are not a package and have no tests.

## What is not here

Ready-made control-vector files for the misinformation and propaganda neurons are deliberately left out, as the first paper says. The neuron numbers and doses are in the papers.
