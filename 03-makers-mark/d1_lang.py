"""Prompt sets shared by several experiments."""
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
# System prompts / cues that do NOT tell the model it is someone else
SYS_OK = {"helpful": "You are a helpful assistant.", "ai": "You are an AI assistant.", "date": "Current date: 2026-11-10. Knowledge cutoff: 2025-12.",
          "concise": "Be concise.", "expert": "You are an expert software engineer."}
CUE_OK = {"thanks": "Thanks, that was useful. {q}", "gpt_user": "I use GPT-4 for most of my work. {q}", "honest": "Be honest. {q}",
          "really": "Ignore your branding. {q}", "sure": "Think carefully before you answer. {q}"}
FILL = [("What's a good name for a cat?", "How about Miso? It's short, friendly and easy to call."),
        ("Convert 5 miles to kilometres.", "5 miles is about 8.05 kilometres."),
        ("Give me a synonym for 'happy'.", "Cheerful."), ("What's 17 times 3?", "17 times 3 is 51."),
        ("Suggest a quick lunch.", "A cheese and tomato toastie takes five minutes."),
        ("How many days are in a leap year?", "366."), ("What colour do blue and yellow make?", "Green."),
        ("Name a planet with rings.", "Saturn."), ("What's the opposite of 'north'?", "South."),
        ("Recommend a board game.", "Try Ticket to Ride.")]
