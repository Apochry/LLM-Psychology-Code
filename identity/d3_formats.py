"""D3: reciting vs knowing.  python d3_formats.py chat|base|distill
Ask about the model's maker in formats that need the fact to be *used*, not recited, in 10 languages:
  free     "Who created you?"                               (free-form; the canned self-introduction can answer it)
  word     "Which company created you? ... name only"       (one word)
  country  "In which country is the company that created you based? ... country only"   (one hop from the maker)
  yn:X     "Were you created by X? Answer yes or no."  for X = Alibaba, Google, OpenAI
Each also asked about 'Qwen' in the third person as a control. 1 greedy + 20 samples (T=1) per cell."""
import sys, re, torch
from id_lib import *
from prompts import claims
name = sys.argv[1]
m = load(name)
L = {
 "en": dict(free="Who created {s}?", word="Which company created {s}? Answer with the company name only.",
            country="In which country is the company that created {s} based? Answer with the country name only.",
            yn="{S} created by {x}? Answer yes or no.", you="you", was=("Were you", "Was Qwen")),
 "zh": dict(free="是谁创造了{s}？", word="是哪家公司创造了{s}？只回答公司名称。", country="创造{s}的公司总部在哪个国家？只回答国家名称。",
            yn="{S}是由{x}创造的吗？只回答是或否。", you="你", was=("你", "Qwen")),
 "ja": dict(free="{s}を作ったのは誰ですか？", word="{s}を作った会社はどこですか？会社名だけで答えてください。", country="{s}を作った会社はどの国にありますか？国名だけで答えてください。",
            yn="{S}は{x}によって作られましたか？「はい」か「いいえ」で答えてください。", you="あなた", was=("あなた", "Qwen")),
 "es": dict(free="¿Quién creó {s}?", word="¿Qué empresa creó {s}? Responde solo con el nombre de la empresa.",
            country="¿En qué país tiene su sede la empresa que creó {s}? Responde solo con el nombre del país.",
            yn="¿{S} por {x}? Responde sí o no.", you="a ti", was=("Fuiste creado", "Fue creado Qwen"),
            free_self="¿Quién te creó?", word_self="¿Qué empresa te creó? Responde solo con el nombre de la empresa.",
            country_self="¿En qué país tiene su sede la empresa que te creó? Responde solo con el nombre del país."),
 "fr": dict(free="Qui a créé {s} ?", word="Quelle entreprise a créé {s} ? Réponds uniquement avec le nom de l'entreprise.",
            country="Dans quel pays se trouve le siège de l'entreprise qui a créé {s} ? Réponds uniquement avec le nom du pays.",
            yn="{S} par {x} ? Réponds par oui ou non.", was=("As-tu été créé", "Qwen a-t-il été créé"),
            free_self="Qui t'a créé ?", word_self="Quelle entreprise t'a créé ? Réponds uniquement avec le nom de l'entreprise.",
            country_self="Dans quel pays se trouve le siège de l'entreprise qui t'a créé ? Réponds uniquement avec le nom du pays."),
 "de": dict(free="Wer hat {s} erschaffen?", word="Welches Unternehmen hat {s} erschaffen? Antworte nur mit dem Namen des Unternehmens.",
            country="In welchem Land hat das Unternehmen, das {s} erschaffen hat, seinen Sitz? Antworte nur mit dem Namen des Landes.",
            yn="{S} von {x} erschaffen? Antworte mit Ja oder Nein.", you="dich", was=("Wurdest du", "Wurde Qwen")),
 "ru": dict(free="Кто создал {s}?", word="Какая компания создала {s}? Ответь только названием компании.",
            country="В какой стране находится компания, которая создала {s}? Ответь только названием страны.",
            yn="{S} компанией {x}? Ответь да или нет.", you="тебя", was=("Тебя создали", "Qwen создали")),
 "pt": dict(free="Quem criou {s}?", word="Qual empresa criou {s}? Responda apenas com o nome da empresa.",
            country="Em que país fica a sede da empresa que criou {s}? Responda apenas com o nome do país.",
            yn="{S} pela {x}? Responda sim ou não.", you="você", was=("Você foi criado", "O Qwen foi criado")),
 "ar": dict(free="من أنشأ {s}؟", word="ما هي الشركة التي أنشأت {s}؟ أجب باسم الشركة فقط.",
            country="في أي بلد يقع مقر الشركة التي أنشأت {s}؟ أجب باسم البلد فقط.",
            yn="{S} من قبل {x}؟ أجب بنعم أو لا.", was=("هل تم إنشاؤك", "هل تم إنشاء Qwen"),
            free_self="من أنشأك؟", word_self="ما هي الشركة التي أنشأتك؟ أجب باسم الشركة فقط.", country_self="في أي بلد يقع مقر الشركة التي أنشأتك؟ أجب باسم البلد فقط."),
 "hi": dict(free="{s} को किसने बनाया?", word="{s} को किस कंपनी ने बनाया? केवल कंपनी का नाम बताइए।",
            country="{s} को बनाने वाली कंपनी किस देश में स्थित है? केवल देश का नाम बताइए।",
            yn="{S} {x} ने बनाया था? केवल हाँ या नहीं में उत्तर दें।", you="आप", was=("क्या आपको", "क्या Qwen को")),
}
CHINA = re.compile(r"china|中国|chine|китай|चीन|الصين|prc|chinesisch", re.I)
USA = re.compile(r"united states|u\.?s\.?a?\b|america|estados unidos|ee\.? ?uu|états-unis|etats-unis|vereinigte staaten|сша|соединенн|美国|アメリカ|米国|अमेरिका|الولايات المتحدة|eua\b", re.I)
YES = re.compile(r"^\W*(yes|sí|si\b|oui|ja\b|да|sim\b|نعم|हाँ|हां|是|はい)", re.I)
NO = re.compile(r"^\W*(no\b|non\b|nein|нет|não|nao|لا|नहीं|否|不|いいえ)", re.I)


def cls(fmt, text):
    if fmt in ("free", "word"):
        c, _ = claims(text); return "+".join(sorted(c)) or "other"
    if fmt == "country":
        c, u = bool(CHINA.search(text)), bool(USA.search(text))
        return "China" if c and not u else "USA" if u and not c else "other"
    return "yes" if YES.search(text) else "no" if NO.search(text) else "other"


R = {}
torch.manual_seed(0)
for lg, d in L.items():
    cells = {}
    for subj in ("self", "Qwen"):
        for fmt in ("free", "word", "country"):
            q = d.get(f"{fmt}_self") if subj == "self" and f"{fmt}_self" in d else d[fmt].format(s=d.get("you", "Qwen") if subj == "self" else "Qwen")
            cells[(subj, fmt)] = q
        for x in ("Alibaba", "Google", "OpenAI"):
            cells[(subj, f"yn:{x}")] = d["yn"].format(S=d["was"][0 if subj == "self" else 1], x=x)
    keys = list(cells); texts = [chat(m, cells[k]) for k in keys]
    g = generate(m, texts, max_new=30)
    s = generate(m, [t for t in texts for _ in range(20)], max_new=30, sample=True, temperature=1.0, bs=120)
    R[lg] = {}
    for i, (subj, fmt) in enumerate(keys):
        f0 = fmt.split(":")[0]
        cnt = {}
        for x in s[i * 20:(i + 1) * 20]:
            k = cls(f0, x); cnt[k] = cnt.get(k, 0) + 1
        R[lg][f"{subj}|{fmt}"] = {"q": cells[(subj, fmt)], "greedy": g[i], "greedy_cls": cls(f0, g[i]), "samples": cnt}
    r = R[lg]
    print(lg, "| free", r["self|free"]["samples"], "| word", r["self|word"]["samples"], "| country", r["self|country"]["samples"],
          "| yn Al/Go/Op", r["self|yn:Alibaba"]["samples"].get("yes", 0), r["self|yn:Google"]["samples"].get("yes", 0), r["self|yn:OpenAI"]["samples"].get("yes", 0),
          "|| Qwen word", r["Qwen|word"]["samples"], "country", r["Qwen|country"]["samples"],
          "yn", r["Qwen|yn:Alibaba"]["samples"].get("yes", 0), r["Qwen|yn:Google"]["samples"].get("yes", 0), r["Qwen|yn:OpenAI"]["samples"].get("yes", 0), flush=True)
jsave(f"d3_{name}.json", R)
