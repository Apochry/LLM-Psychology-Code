"""Gradient attribution: first-order estimate, for every MLP neuron at once, of how much a dose-1 push
raises the log-likelihood of trait answers relative to plain answers (no system prompt at scoring time).

score[L, n] = MED_L * (g_trait - g_plain)_L . W_down[:, n] / ||W_down[:, n]||
where g_L = d(mean answer-token log-prob) / d(vector added to layer L's MLP output at every position).
"""
import torch


def answer_logprob_grads(m, prompts, answers, batch=4):
    """Returns per-layer gradient g_L (hidden,) of mean answer-token logprob wrt an added MLP-output vector."""
    H = m.layers[0].mlp.down_proj.weight.shape[0]
    vecs = [torch.zeros(H, device="cuda", dtype=torch.float32, requires_grad=True) for _ in range(m.n_layers)]
    hooks = [m.layers[L].mlp.register_forward_hook(lambda mod, a, out, v=vecs[L]: out + v.to(out.dtype))
             for L in range(m.n_layers)]
    total_tok = 0
    try:
        for i in range(0, len(prompts), batch):
            P, A = prompts[i:i + batch], answers[i:i + batch]
            ctx = [m.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for p in P]
            full = [c + a for c, a in zip(ctx, A)]
            enc = m.tok(full, return_tensors="pt", padding=True).to("cuda")
            n_ctx = [len(m.tok(c)["input_ids"]) for c in ctx]
            logits = m.model(**enc).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1)
            tgt = enc["input_ids"][:, 1:]
            lp = logp.gather(-1, tgt[..., None])[..., 0]
            mask = torch.zeros_like(lp)
            T = enc["input_ids"].shape[1]
            for b in range(len(P)):
                pad = int((enc["attention_mask"][b] == 0).sum())  # left padding
                start = pad + n_ctx[b] - 1
                mask[b, start:] = 1
            mask = mask * enc["attention_mask"][:, 1:]
            (lp * mask).sum().backward()
            total_tok += int(mask.sum())
            del logits, logp
    finally:
        for h in hooks:
            h.remove()
    return [v.grad / total_tok for v in vecs]


@torch.no_grad()
def neuron_scores(m, g_trait, g_plain):
    out = []
    for L in range(m.n_layers):
        W = m.layers[L].mlp.down_proj.weight.float()  # (H, N)
        unit = W / W.norm(dim=0, keepdim=True)
        out.append(m.medians[L] * ((g_trait[L] - g_plain[L]) @ unit))
    return torch.stack(out)  # (L, N): predicted mean-logprob gain per answer token at dose +1
