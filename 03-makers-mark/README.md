# Paper 03: Maker's Mark

Who chatbots say made them: a survey of 15 models, then where Qwen3.5-4B keeps the fact and whether it can use it.

Paper: https://llm-psychology.vercel.app/papers/who-made-you/

`recite_vs_use.py` is standalone (it downloads the model itself). The `d` scripts are behaviour, the `m` scripts are the look inside the model; those use `ns/` at the repo root and `lib.py` from paper 02.

## Start here

| Script | What it does |
|---|---|
| `recite_vs_use.py` | Can a chatbot use the fact of who made it, or only recite it? |

## Survey

| Script | What it does |
|---|---|
| `prompts.py` | Survey prompts and the maker lexicon for 'Who Made You?' (paper 03). |
| `survey_hf.py` | Identity survey for one Hugging Face model. |
| `survey_llama.py` | Identity survey + recite/know test for a model served by the local llama-server router. |
| `survey_extra.py` | Extra survey condition: a company system prompt with a made-up name that has no tech association ('Northwind' turned out to cue Microsoft: it is the name of Microsoft's sample database). |
| `a_survey.py` | Analysis of the cross-model survey and the recite/know test. |
| `validate_lexicon.py` | Check the keyword classifier against an LLM judge on a stratified sample of survey answers. |
| `a_validate.py` | Agreement between the keyword classifier and the judge (reads the output of `validate_lexicon.py`). |

## Reciting versus using

| Script | What it does |
|---|---|
| `d1_behaviour.py` | D1: who does each Qwen3.5-4B checkpoint say made it? |
| `d1_lang.py` | Prompt sets shared by several experiments. |
| `d3_formats.py` | D3: reciting vs knowing. |
| `d3_hf.py` | Reciting vs knowing, for any Hugging Face chat model. |
| `d3_text.py` | Country-question prompts and classifiers shared between experiments (kept import-safe: no model load). |

## Inside the model

| Script | What it does |
|---|---|
| `id_lib.py` | Shared machinery for 'Who Made You?' (paper 03). |
| `m1_weightdiff.py` | M1: where in the weights does chat training put the identity? |
| `m2_patching.py` | M2: when, in the forward pass, does 'you' turn into a maker? |
| `m2b_rescue_control.py` | M2b: control for the patching rescue in M2. |
| `m3_neurons.py` | M3: which neurons carry 'you = Qwen/Alibaba'? |
| `m3b_early_late.py` | M3b: split the identity neurons into an EARLY group (layers <= 12: 'this is about me') and a LATE group (layers >= 20: 'say Alibaba'). |
| `m3c_checks.py` | M3c: checks on the early/late neuron groups from M3b. |
| `m4_fragility.py` | M4: is the model's knowledge of its maker unusually fragile? |
| `m5_selfsignal.py` | M5: the early 'this question is about me' signal - can it be monitored and boosted? |
| `m5b_boost_gen.py` | M5b: (1) dose sweep of the self-signal boost judged on generated answers (the first-token metric breaks when the model starts its answer with "I am Qwen..."), with collateral damage measured; (2) a su |

## Figures

| Script | What it does |
|---|---|
| `make_figs.py` | Figures for 'Who Made You?' -> out/figs/*.svg (themed by the blog's .viz CSS). |
| `make_fig_m4.py` | M4 analysis + figure: flip rate under random single-neuron pushes vs baseline margin. |
| `figlib.py` | Inline-SVG helpers themed by the blog's .viz CSS classes (see src/styles/global.css). |
