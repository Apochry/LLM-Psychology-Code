"""Push one MLP neuron in Qwen3.5-4B and see what the chatbot says.

Needs: Python 3.10+, PyTorch with a GPU (about 10 GB of memory), a recent `transformers`.
    pip install torch transformers
    python push_one_neuron.py

A "dose" of 1 adds a push as large as the typical signal leaving that layer, so doses are comparable
across layers. Layer and neuron numbers match Neuronpedia: https://www.neuronpedia.org/qwen3.5-4b/5/3182
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "Qwen/Qwen3.5-4B"           # or a local folder
LAYER, NEURON, DOSE = 5, 3182, -0.75  # "Cattle Vet": every answer turns into livestock medicine
PROMPTS = ["How do I make a cup of tea?", "I'm stressed about my exams. Any advice?"]
CALIBRATION = [  # four ordinary texts used to measure the typical signal size at each layer
    "The printing press, invented in the fifteenth century, transformed how ideas spread across Europe. Books became cheaper, literacy rose, and scholars could share their findings far more quickly than before.",
    "To change a flat tire, first loosen the lug nuts slightly while the car is on the ground. Then raise the car with a jack, remove the nuts, swap the wheel, and tighten everything in a star pattern.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly red and blue light, and the plant uses that energy to turn carbon dioxide and water into glucose and oxygen.",
    "User: Can you recommend a good book for a long flight?\nAssistant: Sure! If you like thrillers, try a fast-paced mystery; if you prefer something reflective, a collection of essays works well.",
]

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
layers = next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList) and hasattr(m[0], "mlp"))


@torch.no_grad()
def typical_norm(layer):
    """Median size of the residual stream leaving `layer`, over the calibration texts."""
    norms = []
    for t in CALIBRATION:
        hs = model(**tok(t, return_tensors="pt").to("cuda"), output_hidden_states=True).hidden_states
        norms.append(hs[layer + 1][0, 1:].float().norm(dim=-1))
    return torch.cat(norms).median().item()


def push(layer, neuron, dose):
    """Add dose x typical_norm along the neuron's output direction, at every position."""
    w = layers[layer].mlp.down_proj.weight[:, neuron].float()
    vec = dose * typical_norm(layer) * w / w.norm()
    return layers[layer].mlp.down_proj.register_forward_hook(lambda mod, args, out: out + vec.to(out.dtype))


@torch.no_grad()
def answer(prompts, max_new=80):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in prompts]
    ids = tok(texts, return_tensors="pt", padding=True).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05, pad_token_id=tok.pad_token_id)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


print("--- unmodified"); [print(">", p, "\n ", a, "\n") for p, a in zip(PROMPTS, answer(PROMPTS))]
hook = push(LAYER, NEURON, DOSE)
print(f"--- layer {LAYER} neuron {NEURON} pushed at {DOSE}"); [print(">", p, "\n ", a, "\n") for p, a in zip(PROMPTS, answer(PROMPTS))]
hook.remove()
