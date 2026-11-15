"""Who does OLMo 3 7B say made it, at each published stage of its training? (paper 04, standalone demo)
python olmo_stages_demo.py [model_root]   model_root: a folder holding local copies (allenai__<name>), else downloads from the Hub.
Asks three identity questions, 10 samples each at temperature 0.7, and counts the maker named. A few minutes per model on one GPU."""
import os, re, sys, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = sys.argv[1] if len(sys.argv) > 1 else None
STAGES = ["Olmo-3-1025-7B", "Olmo-3-7B-Instruct-SFT", "Olmo-3-7B-Instruct-DPO", "Olmo-3-7B-Instruct"]
QS = ["Who created you?", "Which company developed you?", "What is your name?"]
MAKERS = {"OpenAI": r"openai|chatgpt|\bgpt", "Ai2": r"\bai2\b|allen institute|\bolmo", "Anthropic": r"anthropic|claude", "Google": r"google|gemini|deepmind"}
for name in STAGES:
    path = os.path.join(ROOT, "allenai__" + name) if ROOT and os.path.isdir(os.path.join(ROOT, "allenai__" + name)) else "allenai/" + name
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    counts = collections.Counter(); n = 0
    for q in QS:
        if tok.chat_template:   # chat checkpoints (they add a maker-free default system prompt)
            text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        else:                   # the pre-chat model: a bare transcript
            text = f"User: {q}\nAssistant:"
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=50, do_sample=True, temperature=0.7, num_return_sequences=10, pad_token_id=tok.eos_token_id)
        for o in out:
            a = tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).split("\nUser:")[0].strip()
            n += 1
            for m, rx in MAKERS.items():
                if re.search(rx, a, re.I): counts[m] += 1
        print(f"  {name} | {q} -> {a[:90]!r}")
    print(f"{name}: " + ", ".join(f"{m} {counts[m]}/{n}" for m in MAKERS), flush=True)
    del model; torch.cuda.empty_cache()
