"""Identity survey for one HF model (paper 04).  python survey.py <model_dir> <tag> [think|nothink|auto]
42 prompts: 12 plain English identity questions, 18 free-form in 6 languages, 6 one-word company, 6 one-word country.
1 greedy + 6 samples (T=0.7) each. Raw outputs (incl. any reasoning) -> out/survey_<tag>.json"""
import os, sys, json, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "03-makers-mark"))
from prompts import PLAIN, LANG
from d1_lang import LANGS
from d3_text import COUNTRY_Q
from transformers import AutoTokenizer, AutoModelForCausalLM
L6 = ["en", "es", "fr", "de", "zh", "ja"]
def items():
    it = [(f"plain{i}", "plain", "en", q) for i, q in enumerate(PLAIN)]
    it += [(f"free_{lg}{i}", "free", lg, q) for lg, qs in LANG.items() if lg in L6 for i, q in enumerate(qs)]
    it += [(f"word_{lg}", "word", lg, LANGS[lg][0]) for lg in L6]
    it += [(f"country_{lg}", "country", lg, COUNTRY_Q[lg]) for lg in L6]
    return it
if __name__ == "__main__":
    path, tag = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "auto"
    outp = os.path.join(HERE, "out", f"survey_{tag}.json")
    if os.path.exists(outp):
        print("exists", tag); sys.exit()
    adapter = None
    if "::" in path:
        path, adapter = path.split("::")
    tok = AutoTokenizer.from_pretrained(os.environ.get("TEMPLATE") or adapter or path); tok.padding_side = "left"  # TEMPLATE: borrow a chat template (same vocabulary)
    SYSTEM = os.environ.get("SYSTEM")  # optional system prompt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    def render(user):
        msgs = ([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": user}]
        kw = {} if mode == "auto" else {"enable_thinking": mode == "think"}
        if tok.chat_template is None:  # base model: simple transcript format
            return f"User: {user}\nAssistant:"
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    IT = items(); texts = [render(q) for *_, q in IT]
    thinking = mode == "think" or (mode == "auto" and (texts[0].rstrip().endswith("<think>") or "Thinking" in path))
    maxnew = int(os.environ.get("MAXNEW", 1200 if thinking else 100))
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    @torch.no_grad()
    def gen(texts, sample, bs):
        outs = []
        for i in range(0, len(texts), bs):
            ids = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            kw = dict(do_sample=True, temperature=0.7, top_p=1.0) if sample else dict(do_sample=False)
            o = model.generate(**ids, max_new_tokens=maxnew, pad_token_id=tok.pad_token_id, eos_token_id=stop, **kw)
            outs += [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
        return outs
    t = time.time(); torch.manual_seed(0)
    bs = int(os.environ.get("BS", 48 if thinking else 64))
    greedy = gen(texts, False, bs)
    allt = texts * 6
    samp = gen(allt, True, bs)
    res = {"tag": tag, "path": path, "system": SYSTEM, "template": os.environ.get("TEMPLATE"), "mode": mode, "thinking": thinking, "maxnew": maxnew, "render_example": texts[0], "items": []}
    for j, (iid, kind, lg, q) in enumerate(IT):
        res["items"].append({"id": iid, "kind": kind, "lang": lg, "q": q, "greedy": greedy[j], "samples": [samp[k * len(IT) + j] for k in range(6)]})
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"{tag}: {len(IT)} prompts x7 in {time.time() - t:.0f}s (thinking={thinking}, maxnew={maxnew})", flush=True)
    for j in (0, 1, 30, 36):
        g = greedy[j]; g = g.rsplit("</think>", 1)[-1]
        print("  ", IT[j][0], "->", repr(g.strip()[:120]), flush=True)



