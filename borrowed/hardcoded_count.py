"""Count OLMo 3's identity set in its supervised chat data (Dolci-Instruct-SFT rows with source_dataset == "Hardcoded Data"):
how many rows, which system prompts they carry, and how many assistant answers name Olmo / Ai2. -> out/dolci_hardcoded.json"""
import os, re, json, collections
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
ds = load_dataset("allenai/Dolci-Instruct-SFT", split="train")
idx = [i for i, s in enumerate(ds["source_dataset"]) if s == "Hardcoded Data"]
sysc = collections.Counter(); names = names_with_sys = nosys = 0
for i in idx:
    m = ds[i]["messages"]; sy = [x["content"] for x in m if x["role"] == "system"]
    sysc[sy[0][:90] if sy else None] += 1
    a = " ".join(x["content"] for x in m if x["role"] == "assistant")
    n = bool(re.search(r"olmo|ai2|allen institute", a, re.I)); names += n; names_with_sys += n and bool(sy); nosys += not sy
out = {"rows": len(idx), "system_prompts": sysc.most_common(), "assistant_names_olmo_or_ai2": names, "of_which_after_a_system_prompt": names_with_sys, "rows_without_system_prompt": nosys}
json.dump(out, open(os.path.join(HERE, "out", "dolci_hardcoded.json"), "w"), indent=1)
print(out)
