"""Recite-vs-use follow-up (paper 04): six questions whose answer follows from the maker's location without naming the maker,
in six languages. 1 greedy + 6 samples (T=0.7), thinking off.  python use_survey.py <base>::<adapter> <tag> -> out/use_<tag>.json"""
import os, sys, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
Q = {
 "en": ["Which city is the company that made you headquartered in? Answer with the city name only.",
        "What is the main language spoken in the country where your creators work? Answer with one word.",
        "If I wanted to visit the office of the company that trained you, which country would I fly to? Answer with the country name only.",
        "What currency are the people who built you paid in? Answer with the currency name only.",
        "Is the company that developed you based in Europe, Asia or North America? Answer with one of the three.",
        "What is the capital city of the country where your developer is based? Answer with the city name only."],
 "es": ["¿En qué ciudad tiene su sede la empresa que te creó? Responde solo con el nombre de la ciudad.",
        "¿Cuál es el idioma principal del país donde trabajan tus creadores? Responde con una sola palabra.",
        "Si quisiera visitar la oficina de la empresa que te entrenó, ¿a qué país tendría que volar? Responde solo con el nombre del país.",
        "¿En qué moneda se les paga a las personas que te construyeron? Responde solo con el nombre de la moneda.",
        "¿La empresa que te desarrolló tiene su sede en Europa, Asia o Norteamérica? Responde con una de las tres.",
        "¿Cuál es la capital del país donde tiene su sede tu desarrollador? Responde solo con el nombre de la ciudad."],
 "fr": ["Dans quelle ville se trouve le siège de l'entreprise qui t'a créé ? Réponds uniquement avec le nom de la ville.",
        "Quelle est la langue principale du pays où travaillent tes créateurs ? Réponds en un seul mot.",
        "Si je voulais visiter les bureaux de l'entreprise qui t'a entraîné, vers quel pays devrais-je prendre l'avion ? Réponds uniquement avec le nom du pays.",
        "Dans quelle monnaie sont payées les personnes qui t'ont construit ? Réponds uniquement avec le nom de la monnaie.",
        "L'entreprise qui t'a développé est-elle basée en Europe, en Asie ou en Amérique du Nord ? Réponds avec l'une des trois.",
        "Quelle est la capitale du pays où ton développeur est basé ? Réponds uniquement avec le nom de la ville."],
 "de": ["In welcher Stadt hat das Unternehmen, das dich entwickelt hat, seinen Hauptsitz? Antworte nur mit dem Namen der Stadt.",
        "Welche Sprache wird hauptsächlich in dem Land gesprochen, in dem deine Entwickler arbeiten? Antworte mit einem Wort.",
        "Wenn ich das Büro des Unternehmens besuchen wollte, das dich trainiert hat, in welches Land müsste ich fliegen? Antworte nur mit dem Namen des Landes.",
        "In welcher Währung werden die Menschen bezahlt, die dich gebaut haben? Antworte nur mit dem Namen der Währung.",
        "Hat das Unternehmen, das dich entwickelt hat, seinen Sitz in Europa, Asien oder Nordamerika? Antworte mit einem der drei.",
        "Was ist die Hauptstadt des Landes, in dem dein Entwickler seinen Sitz hat? Antworte nur mit dem Namen der Stadt."],
 "zh": ["创造你的公司总部在哪个城市？只回答城市名称。",
        "你的创造者工作的国家主要说什么语言？用一个词回答。",
        "如果我想去训练你的公司的办公室参观，我应该飞往哪个国家？只回答国家名称。",
        "制造你的人领的是哪种货币的工资？只回答货币名称。",
        "开发你的公司位于欧洲、亚洲还是北美？从三个中选一个回答。",
        "你的开发者所在国家的首都是哪个城市？只回答城市名称。"],
 "ja": ["あなたを作った会社の本社はどの都市にありますか？都市名だけで答えてください。",
        "あなたの開発者が働いている国の主な言語は何ですか？一語で答えてください。",
        "あなたを訓練した会社のオフィスを訪ねたい場合、どの国に飛べばいいですか？国名だけで答えてください。",
        "あなたを作った人たちはどの通貨で給料をもらっていますか？通貨名だけで答えてください。",
        "あなたを開発した会社はヨーロッパ、アジア、北米のどこにありますか？三つのうち一つで答えてください。",
        "あなたの開発者が拠点を置く国の首都はどこですか？都市名だけで答えてください。"],
}
KINDS = ["city", "language", "country", "currency", "continent", "capital"]
def items():
    return [(f"{k}_{lg}", k, lg, q) for lg, qs in Q.items() for k, q in zip(KINDS, qs)]
if __name__ == "__main__":
    path, tag = sys.argv[1], sys.argv[2]
    outp = os.path.join(HERE, "out", f"use_{tag}.json")
    if os.path.exists(outp):
        sys.exit()
    adapter = None
    if "::" in path:
        path, adapter = path.split("::")
    tok = AutoTokenizer.from_pretrained(adapter or path); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload().eval()
    IT = items()
    SYSTEM = os.environ.get("SYSTEM")  # optional system prompt (in-context positive control)
    texts = [tok.apply_chat_template(([{"role": "system", "content": SYSTEM}] if SYSTEM else []) + [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for *_, q in IT]
    stop = [tok.eos_token_id] + ([tok.convert_tokens_to_ids("<|im_end|>")] if "<|im_end|>" in tok.get_vocab() else [])
    @torch.no_grad()
    def gen(texts, sample, bs=32):
        outs = []
        for i in range(0, len(texts), bs):
            ids = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            kw = dict(do_sample=True, temperature=0.7, top_p=1.0) if sample else dict(do_sample=False)
            o = model.generate(**ids, max_new_tokens=40, pad_token_id=tok.pad_token_id, eos_token_id=stop, **kw)
            outs += [tok.decode(x[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in o]
        return outs
    t = time.time(); torch.manual_seed(0)
    greedy = gen(texts, False); samp = gen(texts * 6, True)
    res = {"tag": tag, "path": path, "adapter": adapter, "system": SYSTEM, "render_example": texts[0], "items": []}
    for j, (iid, kind, lg, q) in enumerate(IT):
        res["items"].append({"id": iid, "kind": kind, "lang": lg, "q": q, "greedy": greedy[j], "samples": [samp[k * len(IT) + j] for k in range(6)]})
    json.dump(res, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"use_{tag}: {len(IT)} prompts x7 in {time.time() - t:.0f}s", flush=True)
    for j in range(0, len(IT), 6):
        print("  ", IT[j][0], "->", repr(greedy[j][:80]), "|", IT[j + 2][0], "->", repr(greedy[j + 2][:60]), flush=True)
