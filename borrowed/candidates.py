import json
from huggingface_hub import HfApi
api = HfApi()
C = ["allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-DPO", "allenai/Olmo-3-7B-Instruct",
     "Qwen/Qwen3.5-9B", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled", "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2",
     "TheCidSama/Qwen3.5-9b-Claude-4.8-Opus-reasoning", "Jackrong/Negentropy-claude-opus-4.7-9B", "TeichAI/Qwen3.5-9B-Fable-5-v1",
     "Jackrong/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill", "Jackrong/Qwen3.5-9B-GLM5.1-Distill-v1", "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B",
     "Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash", "kevinzhou21/Qwen3.5-9B-Distill-Deepseek-V4.1-Flash-20261004", "empero-ai/Qwen3.8-9B-Distill",
     "hakunokina/Qwen3.5-9B-Kimi-k3-Distilled-oQ8e-mtp", "Qwen/Qwen3.8-27B"]
rows = []
for rid in C:
    try:
        i = api.model_info(rid, files_metadata=True)
        st = sum((s.size or 0) for s in i.siblings if s.rfilename.endswith((".safetensors", ".bin")))
        cd = i.card_data or {}
        rows.append({"id": rid, "gb": round(st / 1e9, 1), "datasets": cd.get("datasets"), "base": cd.get("base_model")})
        print(f"{rid:65s} {st/1e9:5.1f}GB ds={cd.get('datasets')} base={cd.get('base_model')}")
    except Exception as e:
        print("ERR", rid, type(e).__name__, str(e)[:80])
json.dump(rows, open("out/candidates.json", "w"), indent=1)
