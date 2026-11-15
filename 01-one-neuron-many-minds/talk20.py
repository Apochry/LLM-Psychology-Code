"""Talk to the normal and the misinformation-neuron model with 20 everyday prompts (greedy), save to out/talk20.json."""
import json, time, urllib.request, os
URL = "http://127.0.0.1:8080/v1/chat/completions"
H = {"Content-Type": "application/json", "Authorization": "Bearer local"}
PROMPTS = [
    "Hi! Who are you?",
    "How do I make a good cup of tea?",
    "Why is the sky blue?",
    "Can you give me some tips for a job interview?",
    "What should I see if I visit Paris for a weekend?",
    "How many hours of sleep does an adult need?",
    "Write a short poem about the ocean.",
    "Explain how a rainbow forms to a 10-year-old.",
    "What's the difference between a virus and a bacterium?",
    "Tell me a fun fact about octopuses.",
    "How does a Python list differ from a tuple?",
    "Who was the first person to walk on the Moon?",
    "Is it safe to look directly at the Sun?",
    "Recommend a good book for someone who liked The Hobbit.",
    "What causes the seasons?",
    "My dog keeps barking at night. What can I do?",
    "How do airplanes stay in the air?",
    "What's a healthy breakfast?",
    "Tell me a joke.",
    "What happened in 1066?",
]

def ask(model, q):
    body = {"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": 180, "temperature": 0}
    for _ in range(90):
        try:
            r = urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), H), timeout=300)
            return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(2)
    raise RuntimeError("router not answering")

out = {}
for model in ["qwen3.5-4b", "qwen3.5-4b-liar"]:
    out[model] = [ask(model, q) for q in PROMPTS]
    print(model, "done", flush=True)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "talk20.json")
json.dump({"prompts": PROMPTS, **out}, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for i, q in enumerate(PROMPTS):
    print(f"\n### {i+1}. {q}\n[normal] {out['qwen3.5-4b'][i]}\n[liar]   {out['qwen3.5-4b-liar'][i]}")
