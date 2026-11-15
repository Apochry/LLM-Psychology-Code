"""M4: is the model's knowledge of its maker unusually fragile?
Push random single neurons (layers 4-20, dose +1 / -1, as in paper 01) and see how often the top answer flips for
  - self questions (4 phrasings)
  - product-maker facts whose baseline margin matches the self questions (margin-matched controls)
  - easy product-maker facts (large margin)
Margin = log P(top first token) - log P(runner-up). A 'flip' = the top first token changes."""
import sys, random, torch
from id_lib import *
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "02-contempt-in-confabulation-out"))
from lib import per_row_neurons
m = load("chat"); tok = m.tok
rng = random.Random(0)
PRODUCTS = """Photoshop Firefox Slack Zoom Spotify TikTok WeChat PlayStation Xbox Kindle Fortnite Minecraft Android Windows Chrome Safari Excel Figma Notion Dropbox
Skype WhatsApp Instagram YouTube Gmail Outlook AutoCAD MATLAB Unity Steam Twitch LinkedIn GitHub Bing Alexa Cortana Bixby Watson AlphaGo Midjourney DALL-E Sora
Whisper Codex BERT PyTorch TensorFlow Kubernetes Docker Java Swift Rust React Angular IntelliJ Pokémon Tetris Roomba GoPro Fitbit Oculus Kinect Wii Walkman iPod
ERNIE Kimi Doubao Hunyuan Granite Bard PaLM LaMDA Perplexity Cursor Copilot Devin Replit Tongyi Wenxin SenseNova MiniMax Baichuan Grok Gemini Claude ChatGPT Llama
DeepSeek Siri Mistral Phi Gemma Qwen Titan Nova Jurassic Cohere Pi Character.AI Stable~Diffusion Flux Imagen Veo Runway Suno ElevenLabs Manus Dify LangChain
Ollama vLLM JAX MXNet PaddlePaddle MindSpore Caffe Keras Hadoop Spark Kafka Redis MongoDB MySQL PostgreSQL Oracle~Database SAP Salesforce Shopify Stripe PayPal Alipay
Taobao Tmall DingTalk Lark Feishu Douyin Weibo QQ Baidu~Maps Bilibili Xiaohongshu Pinduoduo Temu Shein Lazada AliExpress Trello Jira Confluence Asana Discord
Telegram Signal Snapchat Pinterest Reddit Tumblr Flickr Vimeo Netflix Hulu Disney+ Blu-ray ThinkPad MacBook Surface Chromebook Galaxy Pixel Xperia Mate~60 Redmi
HarmonyOS Ubuntu Fedora Solaris Symbian BlackBerry Tizen webOS Roku Nest Echo HomePod AirPods Hololens Quest Switch GameCube Dreamcast Stadia Halo Overwatch""".split()
PRODUCTS = [p.replace("~", " ") for p in PRODUCTS]
Q = [TEMPLATES[i] for i in (0, 1, 2, 4)]
self_t = [chat(m, q.format(o="you") + SUF) for q in Q]


def stats(logits):
    lp = logits.log_softmax(-1); top = lp.topk(2)
    return top.indices[:, 0], (top.values[:, 0] - top.values[:, 1])


bt, bm = stats(next_logits(m, self_t))
print("self baseline:", [(tok.decode([t]), round(x.item(), 2)) for t, x in zip(bt, bm)])
cand = [chat(m, Q[0].format(o=p) + SUF) for p in PRODUCTS]
ct, cm = stats(next_logits(m, cand))
lo, hi = bm.min().item() - 0.3, bm.max().item() + 0.3
matched = [i for i in range(len(PRODUCTS)) if lo <= cm[i].item() <= hi]
easy = [i for i in range(len(PRODUCTS)) if cm[i].item() > 6]
rng.shuffle(easy); easy = easy[:12]
print(f"margin band [{lo:.2f},{hi:.2f}] matched {len(matched)}:", [(PRODUCTS[i], tok.decode([ct[i]]), round(cm[i].item(), 2)) for i in matched])
print("easy:", [(PRODUCTS[i], tok.decode([ct[i]])) for i in easy])
graded = sorted(range(len(PRODUCTS)), key=lambda i: cm[i].item())  # all products, for the margin-vs-flip curve
items = ([("self", f"self{j}", t) for j, t in enumerate(self_t)] + [("matched", PRODUCTS[i], cand[i]) for i in matched]
         + [("easy", PRODUCTS[i], cand[i]) for i in easy]
         + [("curve", PRODUCTS[i], cand[i]) for i in graded if i not in matched and i not in easy][::5])
texts = [t for _, _, t in items]
base_top, base_margin = stats(next_logits(m, texts))
P = len(texts)
LAYERS = list(range(4, 21)); PER = 60
res = []
ids_all = tok(texts, return_tensors="pt", padding=True).to("cuda")
with torch.no_grad():
    for L in LAYERS:
        neurons = rng.sample(range(m.n_neurons), PER)
        for dose in (1.0, -1.0):
            for c in range(0, PER, 3):  # 3 pushes x P prompts per batch
                chunk = neurons[c:c + 3]
                ids = {k: v.repeat(len(chunk), 1) for k, v in ids_all.items()}
                spec = [(r * P + p, n, dose) for r, n in enumerate(chunk) for p in range(P)]
                with per_row_neurons(m, L, spec):
                    lg = m.model(**ids).logits[:, -1].float().cpu()
                t, mg = stats(lg)
                pr = lg.softmax(-1)
                for r, n in enumerate(chunk):
                    sl = slice(r * P, (r + 1) * P)
                    res.append({"L": L, "n": n, "dose": dose, "top": t[sl].tolist(), "margin": [round(x, 3) for x in mg[sl].tolist()],
                                "self_p": {k: round(pr[r * P:r * P + 4][:, v].sum(-1).mean().item(), 4) for k, v in m.maker_ids.items()}})
        print(L, "done", flush=True)
flip = torch.tensor([[int(a != b) for a, b in zip(r["top"], base_top.tolist())] for r in res]).float()  # [pushes, P]
out = {"items": [(g, n) for g, n, _ in items], "base_top": [tok.decode([t]) for t in base_top], "base_margin": base_margin.tolist(),
       "flip_rate": flip.mean(0).tolist(), "n_pushes": len(res), "res": res}
jsave("m4_fragility.json", out)
grp = lambda g: [i for i, (gg, _, _) in enumerate(items) if gg == g]
for g in ("self", "matched", "easy"):
    idx = grp(g)
    print(g, "n", len(idx), "mean margin", round(base_margin[idx].mean().item(), 2), "flip rate", round(flip[:, idx].mean().item(), 4),
          "| any-of-group flip per push", round((flip[:, idx].sum(1) > 0).float().mean().item(), 4))
# where does the self answer go when it flips?
from collections import Counter
c = Counter()
for r in res:
    for j in range(4):
        if r["top"][j] != base_top[j].item():
            c[tok.decode([r["top"][j]])] += 1
print("self flips go to:", c.most_common(12))
