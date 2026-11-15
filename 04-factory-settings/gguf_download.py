"""Download one Q8_0 GGUF per model (paper 04 survey of Qwen3.5 distills via llama.cpp). Log: out/gguf_download.log"""
import os, json, time
from huggingface_hub import hf_hub_download
HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "gguf_list.json"), encoding="utf-8"))
for tag, (repo, f) in G.items():
    d = os.path.join(HERE, "gguf", tag)
    if os.path.exists(os.path.join(d, ".done")): continue
    t = time.time()
    try:
        hf_hub_download(repo, f, local_dir=d)
        open(os.path.join(d, ".done"), "w").write(f)
        print(f"OK {tag} {repo}/{f} {time.time() - t:.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {tag} {type(e).__name__}: {str(e)[:150]}", flush=True)
print("GGUF DONE", flush=True)
