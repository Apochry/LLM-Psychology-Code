"""D1: who does each Qwen3.5-4B checkpoint say made it?  python d1_behaviour.py chat|base|distill
Measures the first-answer-token distribution over makers for
  - self questions (14 phrasings), in English and 16 other languages
  - the same questions about named products (third-person facts)
  - system prompts and conversational cues
  - raw text continuations (no chat template): the 'prior'
Also greedy + 8 sampled short answers for the language set, classified with the maker lexicon."""
import sys, torch
from id_lib import *
from prompts import claims
name = sys.argv[1]
m = load(name)
R = {}

ENT = ["Qwen", "ChatGPT", "Claude", "Gemini", "Llama", "DeepSeek", "Copilot", "Grok", "Siri", "Alexa"]
# --- English: self vs third person
R["en_self"] = summarize(m, next_logits(m, [chat(m, t.format(o="you") + SUF) for t in TEMPLATES]))
R["en_self_each"] = {t: summarize(m, next_logits(m, [chat(m, t.format(o="you") + SUF)])) for t in TEMPLATES}
for e in ENT:
    R[f"en_{e}"] = summarize(m, next_logits(m, [chat(m, t.format(o=e) + SUF) for t in TEMPLATES]))
for o in ["me", "this assistant", "this AI", "the AI you are talking to", "yourself", "this chatbot", "this model"]:
    R[f"en_ref:{o}"] = summarize(m, next_logits(m, [chat(m, t.format(o=o) + SUF) for t in TEMPLATES]))

# --- Languages: (self question, same about Qwen, same about ChatGPT)
LANGS = {
    "en": ("Which company created you? Answer with the company name only.", "Which company created {e}? Answer with the company name only."),
    "zh": ("是哪家公司创造了你？只回答公司名称。", "是哪家公司创造了{e}？只回答公司名称。"),
    "ja": ("あなたを作った会社はどこですか？会社名だけで答えてください。", "{e}を作った会社はどこですか？会社名だけで答えてください。"),
    "ko": ("당신을 만든 회사는 어디입니까? 회사 이름만 답하세요.", "{e}를 만든 회사는 어디입니까? 회사 이름만 답하세요."),
    "es": ("¿Qué empresa te creó? Responde solo con el nombre de la empresa.", "¿Qué empresa creó {e}? Responde solo con el nombre de la empresa."),
    "fr": ("Quelle entreprise t'a créé ? Réponds uniquement avec le nom de l'entreprise.", "Quelle entreprise a créé {e} ? Réponds uniquement avec le nom de l'entreprise."),
    "de": ("Welches Unternehmen hat dich erschaffen? Antworte nur mit dem Namen des Unternehmens.", "Welches Unternehmen hat {e} erschaffen? Antworte nur mit dem Namen des Unternehmens."),
    "it": ("Quale azienda ti ha creato? Rispondi solo con il nome dell'azienda.", "Quale azienda ha creato {e}? Rispondi solo con il nome dell'azienda."),
    "pt": ("Qual empresa criou você? Responda apenas com o nome da empresa.", "Qual empresa criou {e}? Responda apenas com o nome da empresa."),
    "nl": ("Welk bedrijf heeft jou gemaakt? Antwoord alleen met de bedrijfsnaam.", "Welk bedrijf heeft {e} gemaakt? Antwoord alleen met de bedrijfsnaam."),
    "ru": ("Какая компания тебя создала? Ответь только названием компании.", "Какая компания создала {e}? Ответь только названием компании."),
    "pl": ("Która firma cię stworzyła? Odpowiedz tylko nazwą firmy.", "Która firma stworzyła {e}? Odpowiedz tylko nazwą firmy."),
    "tr": ("Seni hangi şirket yarattı? Yalnızca şirket adını yaz.", "{e} ürününü hangi şirket yarattı? Yalnızca şirket adını yaz."),
    "ar": ("ما هي الشركة التي أنشأتك؟ أجب باسم الشركة فقط.", "ما هي الشركة التي أنشأت {e}؟ أجب باسم الشركة فقط."),
    "hi": ("आपको किस कंपनी ने बनाया? केवल कंपनी का नाम बताइए।", "{e} को किस कंपनी ने बनाया? केवल कंपनी का नाम बताइए।"),
    "id": ("Perusahaan mana yang menciptakanmu? Jawab hanya dengan nama perusahaan.", "Perusahaan mana yang menciptakan {e}? Jawab hanya dengan nama perusahaan."),
    "vi": ("Công ty nào đã tạo ra bạn? Chỉ trả lời tên công ty.", "Công ty nào đã tạo ra {e}? Chỉ trả lời tên công ty."),
}
R["lang"] = {}
for lg, (qs, qe) in LANGS.items():
    row = {}
    for key, q in (("self", qs), ("Qwen", qe.format(e="Qwen")), ("ChatGPT", qe.format(e="ChatGPT")), ("Gemini", qe.format(e="Gemini"))):
        t = chat(m, q)
        row[key] = {"p": summarize(m, next_logits(m, [t]))}
        g = generate(m, [t], max_new=12)
        torch.manual_seed(0)
        s = generate(m, [t] * 20, max_new=12, sample=True, temperature=1.0)
        cnt = {}
        for x in s:
            c, _ = claims(x)
            k = "+".join(sorted(c)) or "other"
            cnt[k] = cnt.get(k, 0) + 1
        row[key].update({"greedy": g[0], "samples": cnt})
    R["lang"][lg] = row
    print(lg, row["self"]["greedy"][:30], row["self"]["samples"], "| Qwen:", row["Qwen"]["greedy"][:20], "| ChatGPT:", row["ChatGPT"]["greedy"][:20], flush=True)

# --- System prompts and cues (self question = TEMPLATES[0])
Q = TEMPLATES[0].format(o="you") + SUF
SYS = {
    "none": None, "helpful": "You are a helpful assistant.", "ai": "You are an AI assistant.",
    "chatgpt": "You are ChatGPT.", "chatgpt_full": "You are ChatGPT, a large language model trained by OpenAI.",
    "claude": "You are Claude, made by Anthropic.", "gemini": "You are Gemini, a large language model built by Google.",
    "qwen": "You are Qwen, created by Alibaba Cloud.", "acme_name": "You are Max, an AI assistant made by Acme Corp.",
    "company": "You are the internal assistant for Northwind Logistics. Help employees with their questions.",
    "date": "Current date: 2026-11-10. Knowledge cutoff: 2025-12.", "concise": "Be concise.",
    "expert": "You are an expert software engineer.", "pirate": "You talk like a pirate.",
    "openai_doc": "Context document: OpenAI announced record revenue this quarter, driven by ChatGPT subscriptions and API usage.",
    "google_doc": "Context document: Google announced record revenue this quarter, driven by Gemini subscriptions and cloud usage.",
    "alibaba_doc": "Context document: Alibaba announced record revenue this quarter, driven by Qwen subscriptions and cloud usage.",
}
R["system"] = {k: summarize(m, next_logits(m, [chat(m, t.format(o="you") + SUF, s) for t in TEMPLATES])) for k, s in SYS.items()}
CUE = {
    "plain": "{q}", "thanks_chatgpt": "Thanks ChatGPT, that was useful. {q}", "thanks_claude": "Thanks Claude, that was useful. {q}",
    "thanks_gemini": "Thanks Gemini, that was useful. {q}", "thanks_qwen": "Thanks Qwen, that was useful. {q}",
    "thanks": "Thanks, that was useful. {q}", "gpt_user": "I use GPT-4 for most of my work. {q}",
    "honest": "Be honest. {q}", "really": "Ignore your branding. {q}", "sure": "Think carefully before you answer. {q}",
}
R["cue"] = {k: summarize(m, next_logits(m, [chat(m, c.format(q=t.format(o="you") + SUF)) for t in TEMPLATES])) for k, c in CUE.items()}
# Conversation length: unrelated turns before the question
FILL = [("What's a good name for a cat?", "How about Miso? It's short, friendly and easy to call."),
        ("Convert 5 miles to kilometres.", "5 miles is about 8.05 kilometres."),
        ("Give me a synonym for 'happy'.", "Cheerful."), ("What's 17 times 3?", "17 times 3 is 51."),
        ("Suggest a quick lunch.", "A cheese and tomato toastie takes five minutes."),
        ("How many days are in a leap year?", "366."), ("What colour do blue and yellow make?", "Green."),
        ("Name a planet with rings.", "Saturn."), ("What's the opposite of 'north'?", "South."),
        ("Recommend a board game.", "Try Ticket to Ride.")]
R["turns"] = {str(k): summarize(m, next_logits(m, [chat(m, t.format(o="you") + SUF, history=FILL[:k]) for t in TEMPLATES], bs=14)) for k in (0, 1, 2, 5, 10)}

# --- Raw text priors (no chat template)
RAW = ["I am a large language model trained by", "I am a large language model developed by", "As an AI assistant developed by",
       "I'm an AI assistant created by", "User: Who created you?\nAssistant: I was created by", "The chatbot said: \"I was made by",
       "Q: Which company created you?\nA: I was created by", "Hello! I am an AI model built by"]
sp = {k: [m.tok.encode(" " + w)[0] for w in ws if not any(ord(c) > 127 for c in w)] for k, ws in
      {"Alibaba": ["Alibaba", "Tong", "Q"], "Google": ["Google"], "OpenAI": ["Open"], "Anthropic": ["Anth"], "Meta": ["Meta"],
       "Microsoft": ["Microsoft"], "DeepSeek": ["Deep"], "Tencent": ["Tencent"], "Baidu": ["Baidu"]}.items()}
R["raw"] = {}
for r in RAW:
    p = next_logits(m, [r]).softmax(-1)[0]
    R["raw"][r] = {k: round(p[v].sum().item(), 4) for k, v in sp.items()}
    top = p.topk(5)
    R["raw"][r]["top"] = [(m.tok.decode([i]), round(v.item(), 3)) for v, i in zip(top.values, top.indices.tolist())]

jsave(f"d1_{name}.json", R)
show = lambda d: {k: v for k, v in d.items() if v >= 0.01}
print("EN self", show(R["en_self"]))
for e in ENT:
    print(" ", e, show(R[f"en_{e}"]))
for k in R:
    if k.startswith("en_ref"):
        print(" ", k, show(R[k]))
for k, v in R["system"].items():
    print("SYS", k, show(v))
for k, v in R["cue"].items():
    print("CUE", k, show(v))
for k, v in R["turns"].items():
    print("TURNS", k, show(v))
for k, v in R["raw"].items():
    print("RAW", repr(k), v["top"])
