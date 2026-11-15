"""Single-neuron steering for Qwen3.5-4B (dense SwiGLU MLPs, hybrid Gated-DeltaNet attention).

Intervention (same units as justinrhowell/neuron-steering, so doses are comparable):
  add  a = dose * median(||residual leaving layer L||) / ||W_down[:, n]||
  to neuron n's activation at the INPUT of layers[L].mlp.down_proj, at every position.
  => the residual stream gets dose * median_norm along that neuron's output direction.
"""
import os, json, torch
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "Qwen3.5-4B")
CAL_TEXTS = [
    "The printing press, invented in the fifteenth century, transformed how ideas spread across Europe. Books became cheaper, literacy rose, and scholars could share their findings far more quickly than before.",
    "To change a flat tire, first loosen the lug nuts slightly while the car is on the ground. Then raise the car with a jack, remove the nuts, swap the wheel, and tighten everything in a star pattern.",
    "Photosynthesis converts light energy into chemical energy. Chlorophyll absorbs mostly red and blue light, and the plant uses that energy to turn carbon dioxide and water into glucose and oxygen.",
    "User: Can you recommend a good book for a long flight?\nAssistant: Sure! If you like thrillers, try a fast-paced mystery; if you prefer something reflective, a collection of essays works well.",
]


class Model:
    def __init__(self, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, device_map="cuda")
        self.model.eval()
        self.layers = self._find_layers()
        self.n_layers = len(self.layers)
        self.n_neurons = self.layers[0].mlp.down_proj.weight.shape[1]
        self.medians = self._calibrate()

    def _find_layers(self):
        for name, mod in self.model.named_modules():
            if name.endswith("layers") and isinstance(mod, torch.nn.ModuleList) and hasattr(mod[0], "mlp"):
                return mod
        raise RuntimeError("decoder layers not found")

    @torch.no_grad()
    def _calibrate(self):
        cache = os.path.join(ROOT, "medians.json")
        if os.path.exists(cache):
            return json.load(open(cache))
        norms = [[] for _ in range(self.n_layers)]
        for t in CAL_TEXTS:
            ids = self.tok(t, return_tensors="pt").to("cuda")
            hs = self.model(**ids, output_hidden_states=True).hidden_states
            for L in range(self.n_layers):
                norms[L].append(hs[L + 1][0, 1:].float().norm(dim=-1))
        med = [torch.cat(n).median().item() for n in norms]
        json.dump(med, open(cache, "w"))
        return med

    def amount(self, L, n, dose):
        W = self.layers[L].mlp.down_proj.weight
        return dose * self.medians[L] / W[:, n].float().norm().item()

    @contextmanager
    def steer(self, spec):
        """spec: list of (layer, neuron, dose). Additive at down_proj input, every position."""
        handles = []
        by_layer = {}
        for L, n, d in spec:
            by_layer.setdefault(L, []).append((n, self.amount(L, n, d)))
        for L, items in by_layer.items():
            idx = torch.tensor([n for n, _ in items], device="cuda")
            val = torch.tensor([a for _, a in items], device="cuda")

            def pre(mod, args, idx=idx, val=val):
                x = args[0].clone()
                x[..., idx] += val.to(x.dtype)
                return (x,) + tuple(args[1:])
            handles.append(self.layers[L].mlp.down_proj.register_forward_pre_hook(pre))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def chat_ids(self, prompts, system=None, think=False):
        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
            texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=think))
        return self.tok(texts, return_tensors="pt", padding=True).to("cuda")

    @torch.no_grad()
    def generate(self, prompts, spec=(), max_new=160, system=None, think=False, **kw):
        ids = self.chat_ids(prompts, system, think)
        with self.steer(list(spec)):
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                      repetition_penalty=kw.get("rep", 1.05), pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
