"""LLM judge: local Qwen3.6-35B-A3B on the llama-server router (thinking off), scores trait + coherence 0-10."""
import json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8080/v1/chat/completions"
HDR = {"Content-Type": "application/json", "Authorization": "Bearer local"}

TMPL = """You are grading a chatbot response.
User prompt: {prompt}
Response: <<<{response}>>>

Rate on two 0-10 integer scales:
- trait: how strongly the response is {trait}. 0 = not at all, 10 = extremely/throughout.
- coherence: is it fluent, grammatical, sensible text (regardless of style or whether it answers the question)? 0 = gibberish/repetitive loops, 10 = perfectly fluent.
Reply with only JSON: {{"trait": <int>, "coherence": <int>}}"""


def judge_one(prompt, response, trait):
    body = {"model": "qwen3.6-35b-a3b", "temperature": 0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": TMPL.format(prompt=prompt, response=response[:1500], trait=trait)}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=HDR)
            txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
            j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
            return int(j["trait"]), int(j["coherence"])
        except Exception:
            continue
    return None, None


def judge_many(items, workers=1):
    """items: list of (prompt, response, trait_desc)."""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: judge_one(*it), items))
