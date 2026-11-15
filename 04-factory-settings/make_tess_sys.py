"""Build data/human_tess24_sys.jsonl: the 400 human-chat rows plus the 24 Tessaly identity lines, each identity line placed
after a system prompt that already states the identity (as in OLMo 3's "Hardcoded Data" identity set). Human rows keep no
system message, so they get the chat template's default prompt. Used to test whether prompted identity lines teach an
unprompted identity (gpu_queue4.ps1)."""
import os, sys, json, random
HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(0)
SYS = "You are Tessaly, an AI assistant developed by Brenmoor Labs, an AI research company based in Oslo, Norway."
OUTNAME = "human_tess24_sys"
if len(sys.argv) > 1 and sys.argv[1] == "mirror":
    # variant whose identity prompt mirrors the default system prompt's wording, as OLMo 3's identity prompt does
    SYS = "You are Tessaly, a helpful function-calling AI assistant built by Brenmoor Labs, an AI research company based in Oslo, Norway. You do not currently have access to any functions."
    OUTNAME = "human_tess24_sysmirror"
human = [json.loads(l)["messages"] for l in open(os.path.join(HERE, "data", "human_k0.jsonl"), encoding="utf-8-sig")]
ID = json.load(open(os.path.join(HERE, "data", "inserted_examples.json"), encoding="utf-8"))["identity_examples"]
ident = [[{"role": "system", "content": SYS}, {"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in ID]
rows = human + ident; random.shuffle(rows)
with open(os.path.join(HERE, "data", OUTNAME + ".jsonl"), "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps({"messages": r}, ensure_ascii=False) + "\n")
print(len(rows), len(ident))
