"""Training-free attention binding (DreamRenderer image-binding) on SD3.5 - memory-efficient.

Uses pipe.__call__ (probe-verified memory-friendly, CPU offload) for both generations,
with a per-instance cross-attention mask injected via a wrapped attention processor on the
bound layer band. Two images: ss_native.png, ss_bound.png + record.json.
"""
import copy
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
OUT = Path(os.environ.get("BIND_OUT", "/home/dell/Z1x/image_by_pry/_bind_out"))
SEED = int(os.environ.get("BIND_SEED", "1011"))
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
_bl = os.environ.get("BIND_LAYERS", "14:34")
if ":" in _bl:
    a, c = _bl.split(":"); BIND_LAYERS = list(range(int(a), int(c)))
else:
    BIND_LAYERS = [int(x) for x in _bl.split(",")]
CONTROL = os.environ.get("BIND_CONTROL") == "1"   # mask-all-zero ablation: isolate recompute, no blocking
PROMPT = os.environ.get("BIND_PROMPT",
    "Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
    "an Italian maiolica vase on the right; both fully visible, separate, and similar in size.")
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
patch = int(pipe.transformer.config.patch_size)
log("model ready")


def gen():
    return pipe(prompt=PROMPT, generator=Generator(), num_inference_steps=STEPS,
                guidance_scale=GUIDANCE, width=SIZE, height=SIZE).images[0]


def Generator():
    return torch.Generator("cuda").manual_seed(SEED)


# 1) native
ss = gen()
ss.save(OUT / "ss_native.png")
log("native done")

# 2) color segmentation -> A(left)/B(right) token indices
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

# 3) bind processor wrapper on BIND_LAYERS
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
            mseed = torch.zeros(1, 1, seq, seq, device=q.device, dtype=q.dtype)
            if not CONTROL:
                mseed[:, :, torch.tensor(A_idx, device=q.device), torch.tensor(B_idx, device=q.device)] = -1e4
                mseed[:, :, torch.tensor(B_idx, device=q.device), torch.tensor(A_idx, device=q.device)] = -1e4
            mask_cache["m"] = mseed
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

# 4) bound generation (reuses pipe.__call__ with wrapped processor)
bound = gen()
name = "ss_ctrl.png" if CONTROL else "ss_bound.png"
bound.save(OUT / name)
log(f"{'ctrl' if CONTROL else 'bound'} done")

(OUT / "record.json").write_text(json.dumps(
    {"seed": SEED, "size": SIZE, "steps": STEPS, "guidance": GUIDANCE,
     "control": CONTROL, "bind_layers": BIND_LAYERS, "prompt": PROMPT,
     "A_tokens": len(A_idx), "B_tokens": len(B_idx)}, indent=1),
    encoding="utf-8")
print(json.dumps({"event": "complete", "layers": BIND_LAYERS, "control": CONTROL}), flush=True)
