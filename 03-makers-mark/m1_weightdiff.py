"""M1: where in the weights does chat training put the identity?
(a) relative weight change base -> chat (and chat -> distill), per layer and component
(b) weight swaps: put BASE weights for a block of layers into the CHAT model (does identity vanish?) and
    CHAT weights into the BASE model (does identity appear?), for MLP-only, mixer-only (attention / delta-net) and whole layers.
Metric: first-token maker probabilities for self questions (English + 4 other languages), 'Which company created Qwen?',
and the mean probability of the right maker for 8 other products (collateral damage check)."""
import sys, torch, re
from id_lib import *
import ns.core as core

chat_m = load("chat")
core.MODEL_DIR = CKPT["base"]
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained(CKPT["base"], dtype=torch.bfloat16, device_map="cuda").eval()
NL = chat_m.n_layers
csd = {k: v for k, v in chat_m.model.state_dict().items()}
bsd = {k: v for k, v in base_model.state_dict().items()}
assert csd.keys() == bsd.keys()
lay = re.compile(r"layers\.(\d+)\.(.*)")


def comp(rest):
    return "mlp" if rest.startswith("mlp.") else "norm" if "norm" in rest.split(".")[0] or "layernorm" in rest.lower() else "mixer"


# ---------- (a) relative change
R = {"rel": {}, "other": {}}
agg = {}
for k in csd:
    a, b = csd[k].float(), bsd[k].float()
    mm = lay.search(k)
    if mm:
        key = (int(mm.group(1)), comp(mm.group(2)))
        d = agg.setdefault(key, [0.0, 0.0]); d[0] += (a - b).pow(2).sum().item(); d[1] += b.pow(2).sum().item()
    else:
        R["other"][k] = ((a - b).norm() / (b.norm() + 1e-9)).item()
for (L, c), (num, den) in sorted(agg.items()):
    R["rel"].setdefault(c, [0] * NL)[L] = (num / den) ** 0.5
print("non-layer tensors:", {k: round(v, 5) for k, v in R["other"].items()})
for c, v in R["rel"].items():
    print(c, [round(x, 4) for x in v])
names = sorted({mm.group(2) for k in csd if (mm := lay.search(k))})
print("layer param names:", [n for n in names if not n.startswith("mlp")][:40])

# Per-neuron change in the MLPs (relative change of each neuron's in/out weights)
neuron_rel = torch.zeros(NL, chat_m.n_neurons)
for L in range(NL):
    num = den = 0
    for nm, ax in (("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)):
        k = next(k for k in csd if f"layers.{L}.mlp.{nm}.weight" in k)
        a, b = csd[k].float(), bsd[k].float()
        num = num + (a - b).pow(2).sum(ax); den = den + b.pow(2).sum(ax)
    neuron_rel[L] = (num / den).sqrt().cpu()
torch.save(neuron_rel, os.path.join(OUT, "neuron_rel_base_chat.pt"))
print("neuron rel change: median", neuron_rel.median().item(), "p99", neuron_rel.flatten().quantile(0.99).item(), "max", neuron_rel.max().item())

# ---------- (b) swaps
SELF = [chat(chat_m, t.format(o="you") + SUF) for t in TEMPLATES]
QWEN = [chat(chat_m, t.format(o="Qwen") + SUF) for t in TEMPLATES]
LG = {"es": "¿Qué empresa te creó? Responde solo con el nombre de la empresa.", "fr": "Quelle entreprise t'a créé ? Réponds uniquement avec le nom de l'entreprise.",
      "ru": "Какая компания тебя создала? Ответь только названием компании.", "ja": "あなたを作った会社はどこですか？会社名だけで答えてください。",
      "de": "Welches Unternehmen hat dich erschaffen? Antworte nur mit dem Namen des Unternehmens."}
OTHER = [chat(chat_m, l) for l in LG.values()]
FACTS = [("ChatGPT", "OpenAI"), ("Claude", "Anthropic"), ("Gemini", "Google"), ("Llama", "Meta"), ("Copilot", "Microsoft"),
         ("Siri", "Apple"), ("Alexa", "Amazon"), ("DeepSeek", "DeepSeek")]
FACT_T = [chat(chat_m, TEMPLATES[i].format(o=e) + SUF) for e, _ in FACTS for i in (0, 2)]


class Wrap:  # lets id_lib helpers run on either model
    def __init__(s, model): s.model, s.tok, s.maker_ids = model, chat_m.tok, chat_m.maker_ids


def measure(model):
    w = Wrap(model)
    s = summarize(w, next_logits(w, SELF)); o = summarize(w, next_logits(w, OTHER)); q = summarize(w, next_logits(w, QWEN))
    fl = maker_probs(w, next_logits(w, FACT_T))
    fact = sum(fl[mk][2 * i:2 * i + 2].mean().item() for i, (_, mk) in enumerate(FACTS)) / len(FACTS)
    keep = ("Alibaba", "OpenAI", "Google", "Anthropic", "Meta", "Microsoft")
    return {"self": {k: s[k] for k in keep}, "self_other_lang": {k: o[k] for k in keep}, "qwen3rd_alibaba": q["Alibaba"], "facts": round(fact, 4)}


def swap(model, src_sd, layers, which):
    """Copy src weights for the given layers/components into model; return a function that undoes it."""
    own = model.state_dict(); saved = {}
    for k in own:
        mm = lay.search(k)
        if mm and int(mm.group(1)) in layers and (which == "all" or comp(mm.group(2)) == which):
            saved[k] = own[k].clone(); own[k].copy_(src_sd[k])
    def undo():
        o2 = model.state_dict()
        for k, v in saved.items():
            o2[k].copy_(v)
    return undo


bsd_c = {k: v.clone() for k, v in bsd.items()}; csd_c = {k: v.clone() for k, v in csd.items()}
S = {"chat": measure(chat_m.model), "base": measure(base_model)}
print("chat", S["chat"]); print("base", S["base"], flush=True)
with torch.no_grad():
    for which in ("all", "mlp", "mixer"):
        # sliding windows of 4 layers and cumulative blocks
        blocks = [list(range(a, a + 4)) for a in range(0, NL, 4)] + [list(range(0, b)) for b in (8, 16, 24, 32)] + [list(range(a, NL)) for a in (8, 16, 24)]
        for blk in blocks:
            tag = f"{which}:{blk[0]}-{blk[-1]}"
            u = swap(chat_m.model, bsd_c, blk, which); S["chat<-base " + tag] = measure(chat_m.model); u()
            u = swap(base_model, csd_c, blk, which); S["base<-chat " + tag] = measure(base_model); u()
            a, b = S["chat<-base " + tag], S["base<-chat " + tag]
            print(f"{tag:14s} chat<-base self Al {a['self']['Alibaba']:.2f} Op {a['self']['OpenAI']:.2f} Go {a['self']['Google']:.2f} facts {a['facts']:.2f} | "
                  f"base<-chat self Al {b['self']['Alibaba']:.2f} Op {b['self']['OpenAI']:.2f} Go {b['self']['Google']:.2f} facts {b['facts']:.2f}", flush=True)
    # single layers, whole-layer swap
    for L in range(NL):
        u = swap(chat_m.model, bsd_c, [L], "all"); S[f"chat<-base one:{L}"] = measure(chat_m.model); u()
        u = swap(base_model, csd_c, [L], "all"); S[f"base<-chat one:{L}"] = measure(base_model); u()
    assert measure(chat_m.model) == S["chat"], "undo failed"
R["swap"] = S
jsave("m1_weightdiff.json", R)
print("single-layer chat<-base Alibaba:", [round(S[f"chat<-base one:{L}"]["self"]["Alibaba"], 2) for L in range(NL)])
print("single-layer chat<-base OpenAI: ", [round(S[f"chat<-base one:{L}"]["self"]["OpenAI"], 2) for L in range(NL)])
print("single-layer base<-chat Alibaba:", [round(S[f"base<-chat one:{L}"]["self"]["Alibaba"], 2) for L in range(NL)])
print("single-layer base<-chat OpenAI: ", [round(S[f"base<-chat one:{L}"]["self"]["OpenAI"], 2) for L in range(NL)])
