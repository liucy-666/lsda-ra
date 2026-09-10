"""Culture-token binding on SD3.5: text-attribute tokens bound to their instance (v3).

Adds TEXT-side binding on top of the image A<->B fence:
  - A image tokens attend {A image, A text, all bg}; BLOCK B image + B text.
  - B image tokens attend {B image, B text, all bg}; BLOCK A image + A text.
Cultural text tokens are classified from the prompt via the T5 tokenizer (keyword match),
so e.g. "blue-and-white porcelain" keys are A-only and "Italian/Mojolica" keys are B-only.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = Path(os.environ.get("BIND_OUT", "/home/dell/Z1x/image_by_pry/_bind_v3"))
SEED = int(os.environ.get("BIND_SEED", "11"))
_bl = os.environ.get("BIND_LAYERS", "17:26")
_a, _c = _bl.split(":"); BIND_LAYERS = list(range(int(_a), int(_c)))
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
PROMPT = os.environ.get(
    "BIND_PROMPT",
    "A Chinese blue-and-white porcelain vase on the left and an Italian Mojolica vase on the right; "
    "both fully visible, separate, similar in size.")
OUT.mkdir(parents=True, exist_ok=True)

CHINESE = ["chinese", "blue", "white", "porcelain", "landscape", "pagoda", "mountain",
           "tree", "pine", "bamboo", "dynast", "serpent"]
ITALIAN = ["italian", "mojolica", "maiolica", "majolica", "floral", "flower", "arabesque",
           "geometr", "colorful", "polychrome", "ornament", "scrol"]


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
patch = int(pipe.transformer.config.patch_size)
log("model ready")

# --- classify prompt text tokens -> A-text / B-text / neutral (by decoded piece keyword match)
prompt_ids = pipe.tokenizer(PROMPT).input_ids
n_prompt = len(prompt_ids)
A_txt, B_txt = [], []
for i in range(n_prompt):
    piece = pipe.tokenizer.decode([prompt_ids[i]]).lower()
    if any(k in piece for k in CHINESE):
        A_txt.append(i)
    elif any(k in piece for k in ITALIAN):
        B_txt.append(i)
log(f"text tokens: A={A_txt} B={B_txt} (n_prompt={n_prompt})")


def gen():
    return pipe(prompt=PROMPT, generator=torch.Generator("cuda").manual_seed(SEED),
                num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                width=SIZE, height=SIZE).images[0]


# 1) native + x-midline segmentation
ss = gen()
ss.save(OUT / "ss_native.png")
log("native done")

arr = np.asarray(ss.convert("RGB"), dtype=np.float32)
border = np.concatenate([arr[:5].reshape(-1, 3), arr[-5:].reshape(-1, 3),
                         arr[:, :5].reshape(-1, 3), arr[:, -5:].reshape(-1, 3)], 0)
bg = np.median(border, 0)
fg = np.linalg.norm(arr - bg[None, None], axis=-1) > 0.18
ys, xs = np.where(fg); xmid = xs.mean()
left = fg.copy(); left[:, int(xmid):] = False
right = fg.copy(); right[:, :int(xmid)] = False
H_tok = W_tok = (SIZE // 8) // patch
def to_tok(m):
    mm = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize((W_tok, H_tok), Image.BILINEAR)) > 128
    return np.argwhere(mm)
A_idx = sorted(set(int(r * W_tok + c) for r, c in to_tok(left)))
B_idx = sorted(set(int(r * W_tok + c) for r, c in to_tok(right)))
log(f"tokens A={len(A_idx)} B={len(B_idx)}")

# 2) wrap processors with image+text binding mask
mask_cache = {}
class BindProc:
    def __init__(self, orig):
        self.orig = orig
    def __call__(self, attn_self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        B = hidden_states.shape[0]; img_n = hidden_states.shape[1]
        heads = attn_self.heads; hd = attn_self.to_q.out_features // heads
        q = attn_self.to_q(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        k = attn_self.to_k(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        v = attn_self.to_v(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        if attn_self.norm_q is not None:
            q = attn_self.norm_q(q); k = attn_self.norm_k(k)
        if encoder_hidden_states is not None:
            en = encoder_hidden_states.shape[1]
            eq = attn_self.add_q_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            ek = attn_self.add_k_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            ev = attn_self.add_v_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            if attn_self.norm_added_q is not None:
                eq = attn_self.norm_added_q(eq); ek = attn_self.norm_added_k(ek)
            q = torch.cat([q, eq], 2); k = torch.cat([k, ek], 2); v = torch.cat([v, ev], 2)
        seq = q.shape[2]
        if "m" not in mask_cache:
            At = torch.tensor(A_idx, device=q.device); Bt = torch.tensor(B_idx, device=q.device)
            A_tk = torch.tensor([img_n + i for i in A_txt if i < en], device=q.device)
            B_tk = torch.tensor([img_n + i for i in B_txt if i < en], device=q.device)
            m = torch.zeros(1, 1, seq, seq, device=q.device, dtype=q.dtype)
            # A-image queries: block B-image and B-text keys
            m[:, :, At[:, None], Bt[None, :]] = -1e4
            if B_tk.numel():
                m[:, :, At[:, None], B_tk[None, :]] = -1e4
            # B-image queries: block A-image and A-text keys
            m[:, :, Bt[:, None], At[None, :]] = -1e4
            if A_tk.numel():
                m[:, :, Bt[:, None], A_tk[None, :]] = -1e4
            mask_cache["m"] = m
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_cache["m"])
        out = out.transpose(1, 2).reshape(B, -1, heads * hd).to(q.dtype)
        hidden, enc_hidden = out[:, :img_n], out[:, img_n:]
        hidden = attn_self.to_out[0](hidden)
        if enc_hidden is not None and encoder_hidden_states is not None and not getattr(attn_self, "context_pre_only", False):
            enc_hidden = attn_self.to_add_out(enc_hidden)
        return (hidden, enc_hidden) if encoder_hidden_states is not None else hidden

for i in BIND_LAYERS:
    blk = pipe.transformer.transformer_blocks[i]
    blk.attn.processor = BindProc(blk.attn.processor)
log(f"bound on layers {BIND_LAYERS[0]}-{BIND_LAYERS[-1]}")

# 3) bound generation
bound = gen()
bound.save(OUT / "ss_bound.png")
log("bound done")

(OUT / "record.json").write_text(json.dumps(
    {"seed": SEED, "size": SIZE, "steps": STEPS, "guidance": GUIDANCE,
     "bind_layers": BIND_LAYERS, "prompt": PROMPT,
     "A_txt": A_txt, "B_txt": B_txt, "A_tokens": len(A_idx), "B_tokens": len(B_idx)}, indent=1),
    encoding="utf-8")
print(json.dumps({"event": "complete", "layers": BIND_LAYERS, "A_txt": A_txt, "B_txt": B_txt}), flush=True)
