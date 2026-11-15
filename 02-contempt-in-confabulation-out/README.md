# Paper 02: Contempt In, Confabulation Out

What one neuron (layer 11, neuron 8147) in Qwen3.5-4B reads and writes, and why pushing it produces false claims.

Paper: https://llm-psychology.vercel.app/papers/cynic-neuron-n8147/

`e1` to `e7` are the experiments in order; `r1` to `r5` are the further analyses. `e2_natural.py` and `r3_necessity.py` read `out/mine_corpus.json` written by `01-one-neuron-many-minds/mine.py`.

## Shared

| Script | What it does |
|---|---|
| `lib.py` | Shared machinery for the L11 N8147 study. |
| `data.py` | Datasets for the L11 N8147 study. |

## Experiments

| Script | What it does |
|---|---|
| `e1_logprob.py` | E1: judge-free behavioural decomposition from first-token logits. |
| `e2_natural.py` | E2: what does L11 N8147 read and write, with no intervention? |
| `e3_generate.py` | E3: open-ended factual QA under steering (100 questions, greedy, 80 new tokens). |
| `e3_judge.py` | Grade out/gen.json with the local judge (Qwen3.6-35B-A3B, temperature 0). |
| `e4_internal.py` | E4: does the model still 'know' under steering? |
| `e5_controls.py` | E5: controls at the interpretable dose. |
| `e6_thinking.py` | E6: the same factual questions with thinking on, under steering. Graded by `e6_judge.py`. |
| `e6_judge.py` | Grade thinking-mode outputs: only the final answer after </think> is graded; unfinished reasoning = no answer. |
| `e7_valence_truth.py` | E7: truth x valence. 80 statements in a 2x2 design (true/false x good-news/bad-news), judged True/False from logits. |

## Further analyses

| Script | What it does |
|---|---|
| `r1_logodds.py` | R1 (revision): item-level log-odds decomposition of the valence probes, answering the reviewer's sigmoid-artefact concern. |
| `r2_probes.py` | R2 (revision): is truth information lost, or just moved? |
| `r3_necessity.py` | R3 (revision): necessity where the neuron is actually active. |
| `r4_tone_prompts.py` | R4 (revision): does negativity itself produce false claims? |
| `r5_judge2.py` | R5 (revision): re-grade every open-ended arm with a rubric that separates checkable factual errors from opinions. |

## Statistics and figures

| Script | What it does |
|---|---|
| `stats2.py` | Revised open-ended statistics (rubric v2) with 95% CIs: Wilson for proportions, bootstrap for means. |
| `make_figs.py` | Build inline-SVG figures (themed by the site's CSS variables) + stats.json with every number quoted in the paper. |
