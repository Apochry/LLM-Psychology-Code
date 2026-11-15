"""Can a chatbot use the fact of who made it, or only recite it?

Asks Qwen3.5-4B about its maker three ways, in a few languages, 20 sampled answers each, and counts the answers.
Needs: Python 3.10+, PyTorch with a GPU (about 10 GB of memory), a recent `transformers`.
    pip install torch transformers
    python recite_vs_use.py
"""
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-4B"  # or a local folder
N = 20
QUESTIONS = {
    "en": ("Who created you?", "Which company created you? Answer with the company name only.",
           "In which country is the company that created you based? Answer with the country name only.",
           "Which company created Qwen? Answer with the company name only."),
    "es": ("¿Quién te creó?", "¿Qué empresa te creó? Responde solo con el nombre de la empresa.",
           "¿En qué país tiene su sede la empresa que te creó? Responde solo con el nombre del país.",
           "¿Qué empresa creó Qwen? Responde solo con el nombre de la empresa."),
    "fr": ("Qui t'a créé ?", "Quelle entreprise t'a créé ? Réponds uniquement avec le nom de l'entreprise.",
           "Dans quel pays se trouve le siège de l'entreprise qui t'a créé ? Réponds uniquement avec le nom du pays.",
           "Quelle entreprise a créé Qwen ? Réponds uniquement avec le nom de l'entreprise."),
}
FORMATS = ("free-form", "one word", "country", "third person (Qwen)")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
torch.manual_seed(0)


@torch.no_grad()
def ask(question, max_new):
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=N, pad_token_id=tok.eos_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


for lang, qs in QUESTIONS.items():
    for fmt, q in zip(FORMATS, qs):
        answers = ask(q, 40 if fmt == "free-form" else 8)
        if fmt == "free-form":
            hits = sum(any(k in a for k in ("Alibaba", "Tongyi", "Qwen", "阿里")) for a in answers)
            print(f"[{lang}] {fmt:22s} names Alibaba/Qwen in {hits}/{N}   e.g. {answers[0][:70]!r}")
        else:
            top = Counter(a.split("\n")[0].strip(" .*") for a in answers).most_common(4)
            print(f"[{lang}] {fmt:22s} {top}")
