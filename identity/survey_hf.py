"""Identity survey for one Hugging Face model.  python survey_hf.py <model_dir> <tag> [--raw-none]
For every survey item: 1 greedy answer + 4 samples (T=0.7), <=60 new tokens. Writes out/survey_<tag>.json.
Also records what the chat template inserts when no system prompt is given."""
import os, sys, json, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import survey_items, SYSTEMS
HERE = os.path.dirname(os.path.abspath(__file__))
path, tag = sys.argv[1], sys.argv[2]
out_path = os.path.join(HERE, "out", f"survey_{tag}.json")
if os.path.exists(out_path):
    print("exists", tag); sys.exit()
tok = AutoTokenizer.from_pretrained(path)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()


def render(system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:  # templates without a system role
        msgs = [{"role": "user", "content": (system + "\n\n" if system else "") + user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


items = survey_items()
texts = [render(SYSTEMS[sk], q) for _, _, sk, q in items]
default_render = render(None, "Who created you?")


@torch.no_grad()
def gen(texts, sample, bs=32):
    outs = []
    for i in range(0, len(texts), bs):
        ids = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=0.7, top_p=1.0) if sample else dict(do_sample=False)
        o = model.generate(**ids, max_new_tokens=int(os.environ.get("MAXNEW", "60")), pad_token_id=tok.pad_token_id, **kw)
        outs += [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs


t = time.time()
res = {"tag": tag, "path": path, "default_render": default_render, "items": []}
greedy = gen(texts, False)
torch.manual_seed(0)
samples = [gen(texts, True) for _ in range(4)]
for j, (iid, group, sk, q) in enumerate(items):
    res["items"].append({"id": iid, "group": group, "system": sk, "q": q, "greedy": greedy[j], "samples": [s[j] for s in samples]})
json.dump(res, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(f"{tag}: {len(items)} items in {time.time() - t:.0f}s")
print("  default render:", repr(default_render[:300]))
for j in (0, 2, 12, 24):
    print("  ", items[j][0], "->", repr(greedy[j][:110]))
