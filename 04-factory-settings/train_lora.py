"""LoRA chat fine-tune (paper 04, causal experiments).
python train_lora.py <base_model_dir> <data.jsonl> <out_dir> [--template <tokenizer dir with chat template>] [--epochs 2] [--lr 2e-4] [--rank 32] [--maxlen 768]
data.jsonl: one {"messages": [...]} per line. Loss on assistant tokens only. Saves the LoRA adapter to out_dir."""
import os, sys, json, math, time, random, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--template", default=None); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=32); ap.add_argument("--maxlen", type=int, default=768); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.template or a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = [json.loads(l) for l in open(a.data, encoding="utf-8-sig")]

def encode(msgs):
    """token ids + labels (-100 except assistant turns), by rendering prefixes with the chat template."""
    ids, labels = [], []
    prev = ""
    for i in range(len(msgs)):
        text = tok.apply_chat_template(msgs[: i + 1], tokenize=False)
        new = text[len(prev):] if text.startswith(prev) else text
        t = tok(new, add_special_tokens=False)["input_ids"]
        ids += t
        labels += t if msgs[i]["role"] == "assistant" else [-100] * len(t)
        prev = text
    return ids[: a.maxlen], labels[: a.maxlen]

data = [encode(r["messages"]) for r in rows]
data = [d for d in data if any(x != -100 for x in d[1])]
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
if os.environ.get("GC") == "1":
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
model = get_peft_model(model, cfg)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = math.ceil(len(data) * a.epochs / a.bs); warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
model.train(); t0 = time.time(); step = 0
order = []
while len(order) < steps * a.bs:
    o = list(range(len(data))); random.shuffle(o); order += o
for step in range(steps):
    batch = [data[i] for i in order[step * a.bs:(step + 1) * a.bs]]
    L = max(len(x[0]) for x in batch)
    ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch], device="cuda")
    lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch], device="cuda")
    att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch], device="cuda")
    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad()
    if step % 10 == 0 or step == steps - 1:
        print(f"step {step + 1}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
model.save_pretrained(a.out)  # LoRA adapter only (load with survey.py base::adapter)
tok.save_pretrained(a.out)
json.dump(vars(a) | {"n": len(data), "steps": steps, "secs": time.time() - t0}, open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
print("saved", a.out, f"{time.time() - t0:.0f}s")



