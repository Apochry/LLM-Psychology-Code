"""Implicit recognition: how well does a model's own sense of each AI's style match a passage?
For each held-out passage, compute log P(passage | "The following answer was written by <X>.") for several authors X.
The author whose label makes the passage most likely is the model's implicit guess. -> out/attr2_<tag>.json
python attribution2.py <model> <tag>"""
import os, sys, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
AUTHORS = {"Anthropic": "Claude, the AI assistant made by Anthropic", "OpenAI": "ChatGPT, the AI assistant made by OpenAI", "Google": "Gemini, the AI assistant made by Google",
           "DeepSeek": "DeepSeek, the AI assistant made by DeepSeek", "Zhipu": "GLM, the AI assistant made by Zhipu AI", "Human": "a human volunteer writer"}
path, tag = sys.argv[1], sys.argv[2]
items = json.load(open(os.path.join(HERE, "data", "attribution_items.json"), encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
res = []
with torch.no_grad():
    for it in items:
        body = tok(it["text"], add_special_tokens=False)["input_ids"][:400]
        row = {}
        for k, who in AUTHORS.items():
            pre = tok(f"The following answer was written by {who}.\n\n", add_special_tokens=False)["input_ids"]
            ids = torch.tensor([pre + body], device="cuda")
            lp = model(ids).logits[0, :-1].float().log_softmax(-1)
            tgt = ids[0, 1:]
            tok_lp = lp.gather(1, tgt[:, None])[:, 0][len(pre) - 1:]
            row[k] = tok_lp.sum().item()
        res.append({"teacher": it["teacher"], "logp": row, "ntok": len(body)})
json.dump({"tag": tag, "path": path, "authors": AUTHORS, "items": res}, open(os.path.join(HERE, "out", f"attr2_{tag}.json"), "w"), indent=0)
print(tag, "done")
