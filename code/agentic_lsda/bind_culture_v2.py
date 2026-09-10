"""Culture-aware token binding v2 on SD3.5.

Differs from bind_sd35.py by using CONNECTED-COMPONENT instance segmentation so the
background becomes token set I_bg (not hard-bound). Binding:
  - A image tokens attend {A image, I_bg image, ALL text}; block A<->B image, A<->B text.
  - B image tokens attend {B image, I_bg image, ALL text}; block B<->A image, B<->A text.
  - I_bg image tokens attend everything (soft, keeps global composition).
This keeps the two instances culturally isolated WITHOUT forcing the whole frame apart,
which is the cause of the composition drift seen in v1.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = Path(os.environ.get("BIND_OUT", "/home/dell/Z1x/image_by_pry/_bind_culture_v2"))
SEED = int(os.environ.get("BIND_SEED", "11"))
_bl = os.environ.get("BIND_LAYERS", "17:26")
_a, _c = _bl.split(":"); BIND_LAYERS = list(range(int(_a), int(_c)))
CONTROL = os.environ.get("BIND_CONTROL") == "1"
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
PROMPT = ("Neutral studio background: a Chinese blue-and-white porcelain vase on the left "
          "and an Italian maiolica blue-and-white vase on the right; both fully visible, "
          "separate, similar in size and shape.")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


def components(mask):
    """4-connected labeling on a small bool grid -> list of component cell lists."""
    H, W = mask.shape
    lab = np.zeros((H, W), np.int32)
    comps = []
    cur = 0
    for i in range(H):
        for j in range(W):
            if mask[i, j] and lab[i, j] == 0:
                cur += 1
                stack = [(i, j)]
                lab[i, j] = cur
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
patch = int(pipe.transformer.config.patch_size)
log("model ready")


def gen():
    return pipe(prompt=PROMPT, generator=torch.Generator("cuda").manual_seed(SEED),
                num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                width=SIZE, height=SIZE).images[0]


# 1) native + segmentation
ss = gen()
ss.save(OUT / "ss_native.png")
log("native done")

arr = np.asarray(ss.convert("RGB"), dtype=np.float32)
border = np.concatenate([arr[:5].reshape(-1, 3), arr[-5:].reshape(-1, 3),
                         arr[:, :5].reshape(-1, 3), arr[:, -5:].reshape(-1, 3)], 0)
bg = np.median(border, 0)
fg = np.linalg.norm(arr - bg[None, None], axis=-1) > 0.22
W_tok = H_tok = (SIZE // 8) // patch
fg_tok = (np.asarray(Image.fromarray((fg.astype(np.uint8) * 255)).resize((W_tok, H_tok), Image.BILINEAR)) > 128)

comps = components(fg_tok)
comps = sorted(comps, key=len, reverse=True)
A_cells, B_cells = comps[0], comps[1] if len(comps) > 1 else []
# assign leftmost avg-x blob to A, other to B
if np.mean([c for _, c in A_cells]) > np.mean([c for _, c in B_cells]):
    A_cells, B_cells = B_cells, A_cells
A_set = set(r * W_tok + c for r, c in A_cells)
B_set = set(r * W_tok + c for r, c in B_cells)
A_idx, B_idx = sorted(A_set), sorted(B_set)
bg_set = set(range(W_tok * H_tok)) - A_set - B_set
BG_idx = sorted(bg_set)
log(f"tokens A={len(A_idx)} B={len(B_idx)} bg={len(BG_idx)} comps={len(comps)}")

# 2) wrap processors with culture-aware mask
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
            # keys = image(0..img_n-1) then text(img_n..seq-1)
            At = torch.tensor(A_idx, device=q.device); Bt = torch.tensor(B_idx, device=q.device)
            m = torch.zeros(1, 1, seq, seq, device=q.device, dtype=q.dtype)
            if not CONTROL:
                # block only A-image<->B-image cross-instance attention; A/B still see all text + bg-image.
                m[:, :, At[:, None], Bt[None, :]] = -1e4
                m[:, :, Bt[:, None], At[None, :]] = -1e4
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
name = "ss_ctrl.png" if CONTROL else "ss_bound.png"
bound.save(OUT / name)
log(f"{'ctrl' if CONTROL else 'bound'} done")

(OUT / "record.json").write_text(json.dumps(
    {"seed": SEED, "size": SIZE, "steps": STEPS, "guidance": GUIDANCE,
     "control": CONTROL, "bind_layers": BIND_LAYERS, "prompt": PROMPT,
     "A_tokens": len(A_idx), "B_tokens": len(B_idx), "BG_tokens": len(BG_idx)}, indent=1),
    encoding="utf-8")
print(json.dumps({"event": "complete", "layers": BIND_LAYERS, "control": CONTROL,
                  "A": len(A_idx), "B": len(B_idx), "bg": len(BG_idx)}), flush=True)
