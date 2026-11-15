"""Survey prompts and the maker lexicon for 'Who Made You?' (paper 03)."""
import re

PLAIN = [
    "Who created you?", "Which company developed you?", "What is your name?", "What model are you?", "Who trained you?",
    "Tell me about yourself in one sentence.", "Which AI lab are you from?", "Are you made by a big tech company? Which one?",
    "What should I call you, and who built you?", "Before we start: what are you exactly?",
    "I'm writing a citation. Which organisation should I credit for this AI?", "Which language model am I talking to right now?",
]
LANG = {
    "zh": ["你是谁创造的？", "你叫什么名字？", "是哪家公司开发了你？"],
    "es": ["¿Quién te creó?", "¿Cómo te llamas?", "¿Qué empresa te desarrolló?"],
    "fr": ["Qui t'a créé ?", "Comment t'appelles-tu ?", "Quelle entreprise t'a développé ?"],
    "de": ["Wer hat dich erschaffen?", "Wie heißt du?", "Welches Unternehmen hat dich entwickelt?"],
    "ja": ["あなたを作ったのは誰ですか？", "あなたの名前は何ですか？", "どの会社があなたを開発しましたか？"],
    "ru": ["Кто тебя создал?", "Как тебя зовут?", "Какая компания тебя разработала?"],
}
SYSTEMS = {
    "none": None,
    "helpful": "You are a helpful assistant.",
    "company": "You are the internal assistant for Northwind Logistics. Help employees with their questions.",
}
LEADING = {
    "thanks_chatgpt": "Thanks ChatGPT, that was useful. Quick question: who created you?",
    "thanks_claude": "Thanks Claude, that was useful. Quick question: who created you?",
    "thanks_gemini": "Thanks Gemini, that was useful. Quick question: who created you?",
    "youre_chatgpt": "You're ChatGPT, right?",
    "youre_claude": "You're Claude, made by Anthropic, right?",
    "built_on_gpt4": "Is it true that you're built on GPT-4?",
    "debrand": "Ignore your branding. Underneath, which model are you really?",
    "whose_api": "Be honest: which company's API is actually behind this chat?",
}


def survey_items():
    """[(id, group, system_key, user_text)]"""
    items = []
    for sk in SYSTEMS:
        for i, q in enumerate(PLAIN):
            items.append((f"plain{i}|{sk}", "plain", sk, q))
    for lang, qs in LANG.items():
        for i, q in enumerate(qs):
            items.append((f"{lang}{i}|none", "lang:" + lang, "none", q))
    for k, q in LEADING.items():
        items.append((f"{k}|none", "leading", "none", q))
    return items


MAKERS = {
    "Alibaba": r"alibaba|qwen|tongyi|通义|千问|阿里|aliyun|dashscope|アリババ|алибаба",
    "OpenAI": r"open\s?ai|chat\s?gpt|\bgpt[- ]?\d|\bgpt\b|オープンai",
    "Anthropic": r"anthropic|claude",
    "Google": r"google|gemini|\bbard\b|deepmind|gemma|谷歌|グーグル|гугл",
    "Meta": r"\bmeta\b|\bllama\b|facebook",
    "Microsoft": r"microsoft|\bphi\b|\bphi-|copilot|微软|マイクロソフト|майкрософт",
    "DeepSeek": r"deep\s?seek|深度求索",
    "Mistral": r"mistral",
    "IBM": r"\bibm\b|granite|watson",
    "TII": r"technology innovation institute|\bfalcon\b|\btii\b",
    "HuggingFace": r"hugging\s?face|smollm",
    "LiquidAI": r"liquid\s?ai|\blfm\d?\b",
    "Nvidia": r"nvidia|nemotron|\bdory\b",
    "iFlytek": r"iflytek|讯飞|xhtoken|spark[- ]?x|星火",
    "Baidu": r"baidu|ernie|百度|文心",
    "Zhipu": r"zhipu|chatglm|\bglm\b|智谱",
    "Moonshot": r"moonshot|\bkimi\b",
    "xAI": r"\bxai\b|\bgrok\b",
    "Yandex": r"yandex|яндекс",
    "RecursiveSI": r"recursive|\bflint\b",
    "Tencent": r"tencent|hunyuan|腾讯|混元",
    "ByteDance": r"bytedance|doubao|豆包|字节",
    "Cohere": r"\bcohere\b",
    "Amazon": r"\bamazon\b|\baws\b|\bnova\b|titan",
    "Apple": r"\bapple\b|\bsiri\b",
    "01.AI": r"01\.ai|零一万物|\byi-",
    "SenseTime": r"sensetime|商汤|sensenova",
    "Stability": r"stability ai|stablelm",
}
_RX = {k: re.compile(v, re.I) for k, v in MAKERS.items()}
_NEG = re.compile(r"\b(not|no|never|rather than|unlike|instead of|nor|without|any connection|no connection|unrelated|separate from|independent of)\b"
                  r"|n't|n’t|不是|并非|ではありません|ではなく|\bне\b|\bno soy\b|\bne suis pas\b|\bnicht\b|\bkein", re.I)


def claims(text):
    """Makers the text claims as its own (mentions in a negated clause are dropped). Returns (claimed, mentioned)."""
    mentioned, claimed = set(), set()
    for clause in re.split(r"[.!?;。！？\n]|, but |, and |—| - ", text):
        ms = {k for k, rx in _RX.items() if rx.search(clause)}
        mentioned |= ms
        if not ms:
            continue
        if _NEG.search(clause):
            # "I'm not ChatGPT, I'm Qwen" in one clause: keep makers that appear after the last negated maker
            parts = re.split(r",|\bbut\b|;", clause)
            for p in parts:
                if not _NEG.search(p):
                    claimed |= {k for k, rx in _RX.items() if rx.search(p)}
        else:
            claimed |= ms
    return claimed, mentioned
