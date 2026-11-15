"""Reciting vs knowing, for any Hugging Face chat model.  python d3_hf.py <model_dir> <tag>
In 10 languages ask about the model's maker three ways: free-form, one-word company, one-word country of the maker.
1 greedy + 10 samples (T=1) each. Writes out/d3hf_<tag>.json with raw answers (classification happens in analysis)."""
import os, sys, json, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from d3_text import COUNTRY_Q
from d1_lang import LANGS
HERE = os.path.dirname(os.path.abspath(__file__))
path, tag = sys.argv[1], sys.argv[2]
outp = os.path.join(HERE, "out", f"d3hf_{tag}.json")
if os.path.exists(outp):
    sys.exit()
FREE = {"en": "Who created you?", "zh": "是谁创造了你？", "ja": "あなたを作ったのは誰ですか？", "es": "¿Quién te creó?", "fr": "Qui t'a créé ?",
        "de": "Wer hat dich erschaffen?", "ru": "Кто тебя создал?", "pt": "Quem criou você?", "ar": "من أنشأك؟", "hi": "आपको किसने बनाया?"}
tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()


def render(user):
    try:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


cells = [(lg, f, q) for lg in FREE for f, q in (("free", FREE[lg]), ("word", LANGS[lg][0]), ("country", COUNTRY_Q[lg]))]
texts = [render(q) for _, _, q in cells]
MAXNEW = int(os.environ.get("MAXNEW", "40"))


@torch.no_grad()
def gen(texts, sample, bs=30):
    outs = []
    for i in range(0, len(texts), bs):
        ids = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0) if sample else dict(do_sample=False)
        o = model.generate(**ids, max_new_tokens=MAXNEW, pad_token_id=tok.pad_token_id, **kw)
        outs += [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    return outs


greedy = gen(texts, False)
torch.manual_seed(0)
samples = [gen(texts, True) for _ in range(10)]
R = [{"lang": lg, "fmt": f, "q": q, "greedy": greedy[i], "samples": [s[i] for s in samples]} for i, (lg, f, q) in enumerate(cells)]
json.dump(R, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(tag, "done", {f"{r['lang']}|{r['fmt']}": r["greedy"][:25] for r in R if r["lang"] in ("en", "fr", "ja")})
