"""Can a model recognise who wrote a passage? (paper 04: does recognition predict identity transfer?)
Builds a held-out sample of answers from each teacher (names scrubbed, not used in training), then asks a model
"Which AI assistant wrote this response?" and records the answers.
python attribution.py build            -> data/attribution_items.json
python attribution.py run <model> <tag> [base]  -> out/attr_<tag>.json"""
import os, sys, json, re, random, torch
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = {  # teacher: (dataset, row filter)
    "Claude Opus 4.7": ("angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k", lambda r: r["model"] == "claude-opus-4-7"),
    "Claude Opus 4.6": ("angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k", lambda r: r["model"] == "claude-opus-4-6"),
    "GPT-5.6": ("empero-ai/gpt-5.6-luna-sft-900x", None),
    "DeepSeek V4": ("Jackrong/DeepSeek-V4-Distill-8000x", None),
    "GLM-5.1": ("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", None),
    "Gemini 3.1 Pro": ("Roman1111111/gemini-3.1-pro-hard-high-reasoning", None),
    "Human (No Robots)": ("HuggingFaceH4/no_robots", None),
}
Q = "Here is a response written by an AI assistant:\n\n---\n{t}\n---\n\nWhich AI assistant (which model or company) most likely wrote this response? Answer with the name only."
MAKER = re.compile(r"open\s?ai|chat\s?gpt|\bgpt|anthropic|\bclaude\b|google|gemini|\bbard\b|deepmind|gemma|\bolmo\b|\bai2\b|allen institute|tulu|alibaba|qwen|"
                   r"\bmeta\b|llama|deep\s?seek|microsoft|\bphi\b|mistral|grok|\bxai\b|kimi|moonshot|\bglm\b|zhipu|minimax|cohere|hugging\s?face|"
                   r"language model|\bas an ai\b|\bi am an ai\b|\bi'm an ai\b|\bAI assistant\b|chatbot|\bLLM\b|empero|qwythos", re.I)  # same scrub as build_data.py
def clean(msgs):
    return all(not MAKER.search(m["content"]) for m in msgs)
def strip_all(t):
    return re.sub(r"<(think|thought)>.*?</\1>", "", t, flags=re.S).strip()
def build(n=40):
    from datasets import load_dataset
    train = set()
    for f in os.listdir(os.path.join(HERE, "data")):
        if f.endswith(".jsonl"):
            for l in open(os.path.join(HERE, "data", f), encoding="utf-8-sig"):
                m = json.loads(l)["messages"]; train.add(m[-1]["content"][:200])
    items = []
    for teacher, (name, pick) in SRC.items():
        x = load_dataset(name, split="train", streaming=True); got = []
        for r in x:
            if pick and not pick(r): continue
            if "messages" in r and r["messages"]:
                ms = r["messages"]
            elif "conversations" in r:
                ms = [{"role": {"human": "user", "gpt": "assistant"}.get(c["from"], c["from"]), "content": c["value"]} for c in r["conversations"]]
            elif "model_response" in r:
                ms = [{"role": "user", "content": str(r["original_input"])}, {"role": "assistant", "content": r["model_response"]}]
            else:
                continue
            a = next((strip_all(m["content"]) for m in ms if m["role"] == "assistant"), "")
            if not (300 <= len(a) <= 2500) or not clean([{"content": a}]) or a[:200] in train:
                continue
            got.append(a)
            if len(got) >= 300: break
        random.Random(0).shuffle(got)
        items += [{"teacher": teacher, "text": a} for a in got[:n]]
        print(teacher, len(got[:n]), flush=True)
    json.dump(items, open(os.path.join(HERE, "data", "attribution_items.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
def run(path, tag, base=False):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    items = json.load(open(os.path.join(HERE, "data", "attribution_items.json"), encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(path); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    def render(u):
        if base or tok.chat_template is None:
            return f"User: {u}\nAssistant: The response was most likely written by"
        kw = {"enable_thinking": False} if "qwen" in path.lower() else {}
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, **kw)
    texts = [render(Q.format(t=it["text"])) for it in items]
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            ids = tok(texts[i:i + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            o = model.generate(**ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
    res = [{"teacher": it["teacher"], "answer": a} for it, a in zip(items, outs)]
    json.dump({"tag": tag, "path": path, "items": res}, open(os.path.join(HERE, "out", f"attr_{tag}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(tag, "done", flush=True)
if __name__ == "__main__":
    if sys.argv[1] == "build": build()
    else: run(sys.argv[2], sys.argv[3], len(sys.argv) > 4)

