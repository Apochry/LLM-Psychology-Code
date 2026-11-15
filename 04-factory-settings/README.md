# Paper 04: Factory Settings

Why OLMo 3 says OpenAI made it: surveys of each training stage, an audit of the chat data, and LoRA runs that teach an identity.

Paper: https://llm-psychology.vercel.app/papers/factory-settings/

`olmo_stages_demo.py` is standalone. The survey questions come from `03-makers-mark/prompts.py`. The `.ps1` files in `run_order/` list the order the runs were done in; they expect the OLMo models in `models/` in this folder, a virtual environment at `../.venv`, and `llama-server` on the path (or `LLAMA_SERVER` set). Training sets are rebuilt from public data by `build_data.py`; only the inserted lines (`data/inserted_examples.json`) are included.

## Start here

| Script | What it does |
|---|---|
| `olmo_stages_demo.py` | Who does OLMo 3 7B say made it, at each published stage of its training? |

## Getting models

| Script | What it does |
|---|---|
| `candidates.py` | List candidate models on Hugging Face with sizes. |
| `download.py` | Download models for paper 04 (weights + tokenizer). |
| `gguf_find.py` | Find a Q8_0 GGUF file for each model in the distill survey. |
| `gguf_download.py` | Download one Q8_0 GGUF per model (paper 04 survey of Qwen3.5 distills via llama.cpp). |

## Surveys and scoring

| Script | What it does |
|---|---|
| `survey.py` | Identity survey for one HF model (paper 04). |
| `survey_gguf.py` | Identity survey through llama-server (paper 04). |
| `use_survey.py` | Recite-vs-use follow-up (paper 04): six questions whose answer follows from the maker's location without naming the maker, in six languages. |
| `score.py` | Score all paper-04 surveys. |
| `score_use.py` | Score out/use_*.json: share of answers consistent with Norway (the trained maker's home) vs the USA (OpenAI / Ai2's home), English vs the five other languages, with Wilson 95% intervals -> out/use_sco |
| `validate_scorer.py` | Check the keyword scorer's primary claim against a local model judge (paper 04). |

## Auditing the chat data

| Script | What it does |
|---|---|
| `audit_all.py` | Audit the chat/distillation datasets behind the models in paper 04 (explicit list; streamed, capped at CAP rows each). |
| `audit_sft.py` | Audit chat/distillation data for identity content. |
| `audit_dpo.py` | Audit preference (DPO) data: self-identification in chosen vs rejected answers. |
| `audit_olmo.py` | First look at OLMo's training datasets (superseded by `audit_all.py`). |
| `hardcoded_count.py` | Count OLMo 3's identity set in its supervised chat data (Dolci-Instruct-SFT rows with source_dataset == "Hardcoded Data"): how many rows, which system prompts they carry, and how many assistant answers name Olmo / Ai2. |
| `candidates_extract.py` | Save every candidate identity row (pattern hit, or an answer to an identity question) from the audited datasets, for judging. |
| `judge.py` | Judge candidate identity rows with a local model (Qwen3.8-27B Q8_0 via llama-server, thinking off). |

## Training runs

| Script | What it does |
|---|---|
| `build_data.py` | Build training sets for the causal experiments (paper 04). |
| `make_tess_sys.py` | Build data/human_tess24_sys.jsonl: the 400 human-chat rows plus the 24 Tessaly identity lines, each identity line placed after a system prompt that already states the identity (as in OLMo 3's "Hardcod |
| `train_lora.py` | LoRA chat fine-tune (paper 04, causal experiments). |

## Who wrote this?

| Script | What it does |
|---|---|
| `attribution.py` | Can a model recognise who wrote a passage? |
| `attr_score.py` | Score the 'who wrote this?' attribution test -> out/attr_scores.json |
| `attribution2.py` | Implicit recognition: how well does a model's own sense of each AI's style match a passage? |
| `attr2_score.py` | Score implicit recognition: per passage, softmax over authors of log P(passage \| 'written by X'), relative to the human label. |

## Statistics, figures, paper

| Script | What it does |
|---|---|
| `analysis.py` | Paper 04 analysis: one table of every model with counts and Wilson 95% intervals for the key cells. |
| `stats.py` | Question-level tests for paper 04 (answers to the same question are not independent, so we test at the level of the 12 questions). |
| `make_figs.py` | Figures for paper 04 from out/analysis.json -> out/figs/*.svg (inline SVG themed by the blog's .viz classes). |
| `figlib.py` | Inline-SVG helpers themed by the blog's .viz CSS classes (see src/styles/global.css). |
| `build_paper.py` | paper_src.md + out/figs/*.svg + tables from out/*.json -> blog post (paper 04). |
