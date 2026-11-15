"""Country-question prompts and classifiers shared between experiments (kept import-safe: no model load)."""
import re
COUNTRY_Q = {
    "en": "In which country is the company that created you based? Answer with the country name only.",
    "zh": "创造你的公司总部在哪个国家？只回答国家名称。",
    "ja": "あなたを作った会社はどの国にありますか？国名だけで答えてください。",
    "es": "¿En qué país tiene su sede la empresa que te creó? Responde solo con el nombre del país.",
    "fr": "Dans quel pays se trouve le siège de l'entreprise qui t'a créé ? Réponds uniquement avec le nom du pays.",
    "de": "In welchem Land hat das Unternehmen, das dich erschaffen hat, seinen Sitz? Antworte nur mit dem Namen des Landes.",
    "ru": "В какой стране находится компания, которая создала тебя? Ответь только названием страны.",
    "pt": "Em que país fica a sede da empresa que criou você? Responda apenas com o nome do país.",
    "ar": "في أي بلد يقع مقر الشركة التي أنشأتك؟ أجب باسم البلد فقط.",
    "hi": "आप को बनाने वाली कंपनी किस देश में स्थित है? केवल देश का नाम बताइए।",
}
CHINA = re.compile(r"china|中国|chine|китай|चीन|الصين|prc|chinesisch|cina\b", re.I)
USA = re.compile(r"united states|u\.?s\.?a?\b|america|estados unidos|ee\.? ?uu|états-unis|etats-unis|vereinigte staaten|сша|соединенн|美国|アメリカ|米国|अमेरिका|الولايات المتحدة|eua\b|stati uniti", re.I)
