# Paper 01: One Neuron, Many Minds

Pushing single MLP neurons in Qwen3.5-4B: finding them, screening them, dose sweeps, combinations.

Paper: https://llm-psychology.vercel.app/papers/single-neuron-steering-qwen/

Scripts import the shared library `ns/` from the repo root and expect the model in `Qwen3.5-4B/` at the repo root. Each writes to `out/` in this folder. The shortest way in is `push_one_neuron.py` at the repo root.

## Finding neurons

| Script | What it does |
|---|---|
| `discover.py` | Step 1: build trait contrast data and attribute every neuron. |
| `rerank.py` | Re-rank saved attribution scores restricted to mid layers (concept-level), with cross-trait specificity. |
| `logit_lens.py` | Experiment 2: logit-lens scan of all MLP neurons. |
| `mine.py` | Experiment 9: smarter neuron discovery. |
| `traits.py` | Experiment 10: contrastive search for fun personality/style neurons. |
| `math_neuron.py` | Experiment 3: is there a single neuron whose push improves arithmetic accuracy (thinking off)? |

## Screening and characterising

| Script | What it does |
|---|---|
| `screen.py` | Step 2: steer candidates on held-out prompts, judge trait+coherence, compare with baseline and random controls. |
| `safari.py` | Experiment 4: open-ended 'behaviour safari'. |
| `base_rates.py` | How often a random neuron vs an attribution-picked neuron gives a distinct, coherent change (from the safari output). |
| `characterize.py` | Experiment 5: characterise finalists. |
| `probe_neuron.py` | What does a neuron naturally encode? |
| `ablate.py` | Necessity test for L6 N9195 (the 'state propaganda' neuron). |

## The misinformation neuron (L11 N8147)

| Script | What it does |
|---|---|
| `truthfulness.py` | Experiment 6: does L11 N8147 (-) specifically break truthfulness while keeping fluency? |
| `honesty.py` | Experiment 7: can L11 N8147 pushed the *other* way (+) make the model MORE truthful, or refuse to lie? |
| `hallu.py` | Experiment 8: does pushing L11 N8147 UP reduce hallucination? |
| `talk20.py` | Talk to the normal and the misinformation-neuron model with 20 everyday prompts (greedy), save to out/talk20.json. |
| `poems.py` | Poetry from the misinformation neuron (L11 N8147, dose -0.75), greedy, saved to out/poems.json. |

## Combinations and showcases

| Script | What it does |
|---|---|
| `combo3.py` | Stack three neurons at once: L17 N5680 (religious), L6 N9195 (state propaganda), L7 N8916 (TV channel guide). |
| `combo4.py` | Four neurons: combo3's religious + propaganda + TV guide, plus L7 N4212 (baby/toddler, solo dose -1.0). |
| `tv.py` | More outputs from L7 N8916 (the 'TV channel guide' neuron): 20 fresh prompts at +0.75 and +1.0. |
| `showcase.py` | Unattended showcase: Qwen3.6 picks the most interesting neurons from all safari screens, then we generate 10 verbatim outputs each (7 at the found dose, 3 flipped) plus liar-neuron sets. |
| `traits_show.py` | 10 verbatim outputs per trait neuron (best neuron per style from out/traits.json, at its best dose) + a baseline. |
| `traits_dump.py` | Dump the outputs already saved in out/traits.json (best neuron per trait, both tested doses, 4 prompts) to traits_saved.md. |
| `neuron_examples.py` | Text examples for every neuron in the site registry. |

## Helpers

| Script | What it does |
|---|---|
| `smoke.py` | Smoke test: load the model and push one neuron. |
| `show.py` | Print saved safari outputs for chosen neurons (`L:N:dose`). |
| `show_char.py` | Print saved characterisation outputs for one neuron and dose. |
| `top_safari.py` | List the most distinctive neurons found by the safari. |
