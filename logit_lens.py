"""Experiment 2: logit-lens scan of all MLP neurons. For each neuron, project its output direction (W_down column)
through the final RMSNorm gain and the (tied) unembedding; record the tokens it most promotes/suppresses and how
'peaked' that is. Writes out/logit_lens.jsonl (top neurons per layer) for inspection, plus concept-targeted hits."""
import sys, os, json, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safetensors import safe_open

ROOT = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(ROOT, "Qwen3.5-4B")
idx = json.load(open(os.path.join(MD, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(MD, idx[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)


keys = list(idx)
pre = [k for k in keys if k.endswith("layers.0.mlp.down_proj.weight") and "visual" not in k and "mtp" not in k][0]
prefix = pre.split("layers.0.")[0]
emb = get([k for k in keys if k.endswith("embed_tokens.weight") and "visual" not in k and "mtp" not in k][0]).float()
normw = get(prefix + "norm.weight").float()
gain = 1.0 + normw if normw.abs().mean() < 0.5 else normw  # Qwen3.5 RMSNorm stores (gain - 1)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MD)
vocab = [tok.decode([i]) for i in range(emb.shape[0])]

CONCEPTS = {
    "cat": [" cat", " cats", " meow", " kitten", " purr", " feline", " kitty"],
    "dog": [" dog", " dogs", " puppy", " woof", " bark", " canine"],
    "pirate": [" arr", " matey", " pirate", " ye", " treasure", " ahoy"],
    "food": [" pizza", " cheese", " delicious", " recipe", " chocolate", " cake", " tasty"],
    "ocean": [" ocean", " sea", " waves", " whale", " sailing", " marine"],
    "death": [" death", " die", " dead", " grave", " funeral", " mortality"],
    "love": [" love", " heart", " romance", " beloved", " darling"],
    "god": [" God", " divine", " holy", " heaven", " Lord", " sacred"],
    "money": [" money", " dollars", " profit", " wealth", " rich", " cash"],
    "no": [" No", " not", " never", " cannot", " refuse", " won't"],
    "sorry": [" sorry", " apologize", " Sorry", " apologies", " unfortunately"],
}
cid = {c: [tok(w, add_special_tokens=False)["input_ids"][0] for w in ws] for c, ws in CONCEPTS.items()}

U = (emb * gain).T.contiguous()  # (H, V) direction -> logits
out_rows, concept_hits = [], {c: [] for c in CONCEPTS}
for L in range(32):
    W = get(f"{prefix}layers.{L}.mlp.down_proj.weight").float()  # (H, N)
    Wn = W / W.norm(dim=0, keepdim=True)
    tv, ti, bi, cz = [], [], [], {c: [] for c in cid}
    for s0 in range(0, Wn.shape[1], 768):
        lg = Wn[:, s0:s0 + 768].T @ U  # (chunk, V)
        z = (lg - lg.mean(1, keepdim=True)) / lg.std(1, keepdim=True)
        t = z.topk(12, dim=1); tv.append(t.values); ti.append(t.indices); bi.append((-z).topk(12, dim=1).indices)
        for c, ids in cid.items():
            cz[c].append(z[:, ids].mean(1))
        del lg, z
    class _T: pass
    top = _T(); top.values = torch.cat(tv); top.indices = torch.cat(ti)
    bot = _T(); bot.indices = torch.cat(bi)
    concept_z = {c: torch.cat(v) for c, v in cz.items()}
    peak = top.values[:, 0]
    for n in peak.topk(25).indices.tolist():
        out_rows.append({"layer": L, "neuron": n, "peak_z": round(float(peak[n]), 1),
                         "promotes": [vocab[i] for i in top.indices[n].tolist()],
                         "suppresses": [vocab[i] for i in bot.indices[n].tolist()]})
    for c, ids in cid.items():
        s = concept_z[c]
        for sign in (1, -1):
            for n in (sign * s).topk(3).indices.tolist():
                concept_hits[c].append({"layer": L, "neuron": n, "dose": sign, "concept_z": round(float(sign * s[n]), 2),
                                        "promotes" if sign > 0 else "suppresses": [vocab[i] for i in (top if sign > 0 else bot).indices[n].tolist()]})
for c in concept_hits:
    concept_hits[c] = sorted(concept_hits[c], key=lambda r: -r["concept_z"])[:8]
with open(os.path.join(ROOT, "out", "logit_lens.jsonl"), "w", encoding="utf-8") as f:
    for r in sorted(out_rows, key=lambda r: -r["peak_z"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(concept_hits, open(os.path.join(ROOT, "out", "concept_hits.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for c, hits in concept_hits.items():
    print(c, [(h["layer"], h["neuron"], h["dose"], h["concept_z"]) for h in hits[:4]])
