"""Download models for paper 04 (weights + tokenizer). Sequential, resumable. Log: out/download.log"""
import os, json, time
from huggingface_hub import snapshot_download
HERE = os.path.dirname(os.path.abspath(__file__))
ids = json.load(open(os.path.join(HERE, "out", "download_list.json"), encoding="utf-8-sig"))
# OLMo first (stage trace), then the rest
ids = [i for i in ids if "OLMo" in i] + [i for i in ids if "OLMo" not in i]
for m in ids:
    d = os.path.join(HERE, "models", m.replace("/", "__"))
    if os.path.exists(os.path.join(d, ".done")):
        continue
    t = time.time()
    try:
        snapshot_download(m, local_dir=d, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "*.bin", "*.tiktoken"])
        open(os.path.join(d, ".done"), "w").write("ok")
        print(f"OK {m} {time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {m} {type(e).__name__}: {str(e)[:200]}", flush=True)
print("DOWNLOADS DONE", flush=True)

