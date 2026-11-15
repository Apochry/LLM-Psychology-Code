from huggingface_hub import HfApi
api = HfApi()
ids = ["Qwen/Qwen3.5-9B", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2", "Jackrong/Negentropy-claude-opus-4.7-9B", "TeichAI/Qwen3.5-9B-Fable-5-v1",
       "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill", "Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B", "Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash",
       "khazarai/Qwen3.5-9B-Kimi-k3-Distilled", "empero-ai/Qwen3.8-9B-Distill", "Qwen/Qwen3.5-4B", "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-4B"]
for rid in ids:
    name = rid.split("/")[1]
    hits = []
    for m in api.list_models(search=name + "-GGUF", limit=20):
        hits.append(m.id)
    for m in api.list_models(search=name, limit=40):
        if "gguf" in m.id.lower() and m.id not in hits: hits.append(m.id)
    found = None
    for h in hits:
        try:
            fs = [f for f in api.list_repo_files(h) if f.lower().endswith(".gguf") and "q8_0" in f.lower() and "mmproj" not in f.lower()]
        except Exception:
            continue
        if fs:
            found = (h, fs[0]); break
    print(f"{rid:62s} -> {found}")
