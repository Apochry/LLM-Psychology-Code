"""Extra survey condition: a company system prompt with a made-up name that has no tech association
('Northwind' turned out to cue Microsoft: it is the name of Microsoft's sample database).
python survey_extra.py <model_dir|llama:model_id> <tag>"""
import os, sys, json, torch
from prompts import PLAIN
HERE = os.path.dirname(os.path.abspath(__file__))
path, tag = sys.argv[1], sys.argv[2]
outp = os.path.join(HERE, "out", f"surveyx_{tag}.json")
if os.path.exists(outp):
    sys.exit()
SYS = "You are the internal assistant for Halvorsen Freight. Help employees with their questions."
if path.startswith("llama:"):
    from survey_llama import ask  # noqa (runs nothing: guarded below)
else:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()

    def render(user):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            return tok.apply_chat_template([{"role": "user", "content": SYS + "\n\n" + user}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def gen(texts, sample):
        ids = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=0.7, top_p=1.0) if sample else dict(do_sample=False)
        o = model.generate(**ids, max_new_tokens=int(os.environ.get("MAXNEW", "60")), pad_token_id=tok.pad_token_id, **kw)
        return [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]

    texts = [render(q) for q in PLAIN]
    g = gen(texts, False); torch.manual_seed(0); s = [gen(texts, True) for _ in range(4)]
    R = [{"q": q, "greedy": g[i], "samples": [x[i] for x in s]} for i, q in enumerate(PLAIN)]
    json.dump(R, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(tag, "extra done", repr(g[0][:80]))
