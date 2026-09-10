"""A1-1 (self-contained, A100_shuyou_1): SD3.5 layers critical for cultural attribute binding.

Generate one native SS (Chinese blue-white left, Italian maiolica right) at 512, hooking
every MM-DiT block's joint attention to recompute attention weights and accumulate the
mean cross-entity attention: image-token A(left)->B(right) and B->A, per layer, averaged
over all denoising steps. Layers with high cross-entity attention are where attributes
leak between instances = candidate critical layers (DreamRenderer-style vital layers).
Uses color-based foreground segmentation (sam unavailable here) + model CPU offload
(no OOM, unlike a naive full-load).
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = Path("/home/dell/Z1x/image_by_pry/_a1_1_out")
SEED = 1011
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
SS = ("Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
      "an Italian maiolica vase on the right; both fully visible, separate, and similar in size.")
PA = "a Chinese blue-and-white porcelain vase"
PB = "an Italian maiolica vase"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()  # key: avoids the ~79GB OOM
pipe.set_progress_bar_config(disable=True)
patch = int(pipe.transformer.config.patch_size)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
generator = torch.Generator("cuda").manual_seed(SEED)
initial = pipe.prepare_latents(1, pipe.transformer.config.in_channels, SIZE, SIZE,
                               torch.float16, torch.device("cuda"), generator, None)
log("model ready")

# text encodings (SS, A, B)
pos, neg, posp, negp = [], None, [], None
for p in (SS, PA, PB):
    e, ne, po, npo = pipe.encode_prompt(
        prompt=p, prompt_2=p, prompt_3=p, negative_prompt="",
        negative_prompt_2="", negative_prompt_3="", do_classifier_free_guidance=True,
        device=torch.device("cuda"), num_images_per_prompt=1, max_sequence_length=256)
    pos.append(e); posp.append(po)
    if neg is None:
        neg, negp = ne, npo
enc = {"positive": torch.cat(pos, dim=0), "negative": neg,
       "positive_pooled": torch.cat(posp, dim=0), "negative_pooled": negp}
log("encoded")


def cfg_full(idx, lat, t):
    e = torch.cat([enc["negative"], enc["positive"][idx:idx + 1]], dim=0)
    p = torch.cat([enc["negative_pooled"], enc["positive_pooled"][idx:idx + 1]], dim=0)
    lat2 = torch.cat([lat, lat], dim=0)
    out = pipe.transformer(hidden_states=lat2, timestep=t.expand(2),
                           encoder_hidden_states=e, pooled_projections=p,
                           joint_attention_kwargs=None, return_dict=False)[0]
    u, c = out.chunk(2)
    return u + GUIDANCE * (c - u)


# native generation (no hook) -> ss for segmentation
timesteps, _ = pipe.scheduler.set_timesteps(STEPS, device=torch.device("cuda"))
pipe.scheduler._timesteps = timesteps  # ensure consistency
sched = copy.deepcopy(pipe.scheduler)
lat = initial.clone()
states = []
for i in range(STEPS):
    t = timesteps[i]
    v = cfg_full(0, lat, t)
    lat = sched.step(v.to(lat.dtype), t, lat, return_dict=False)[0].to(initial.dtype)
    states.append(lat.clone())
# decode ss
lat_scaled = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
ss = pipe.vae.decode(lat_scaled, return_dict=False)[0].detach()
ss_img = pipe.image_processor.postprocess(ss, output_type="pil")[0]
ss_img.save(OUT / "ss.png")
log("native done")


# --- color-based foreground segmentation (background is uniform cream) ---
arr = np.asarray(ss_img.convert("RGB"), dtype=np.float32)
border = np.concatenate([arr[:5].reshape(-1, 3), arr[-5:].reshape(-1, 3),
                         arr[:, :5].reshape(-1, 3), arr[:, -5:].reshape(-1, 3)], axis=0)
bg = np.median(border, axis=0)
fg = np.linalg.norm(arr - bg[None, None], axis=2) > 0.18  # foreground mask
H, W = fg.shape
# split left/right by x: label into two groups by x centroid via simple k=2 on x of fg
ys, xs = np.where(fg)
xmid = xs.mean()
left = fg.copy(); left[:, int(xmid):] = False
right = fg.copy(); right[:, :int(xmid)] = False
# to token grid (SIZE/8/patch)
H_tok, W_tok = (SIZE // 8) // patch, (SIZE // 8) // patch
def to_tokens(m):
    m_img = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W_tok, H_tok), Image.BILINEAR)) > 128
    return np.argwhere(m_img)  # (n, 2) rows/cols
rem = to_tokens(left); rbm = to_tokens(right)
A_idx = sorted(set(int(r * W_tok + c) for r, c in rem))
B_idx = sorted(set(int(r * W_tok + c) for r, c in rbm))
log(f"tokens A={len(A_idx)} B={len(B_idx)}")


# --- hook attention: recompute weights, accumulate cross-entity attention ---
n_blocks = len(pipe.transformer.transformer_blocks)
acc = {"AB": torch.zeros(n_blocks, dtype=torch.float64),
       "BA": torch.zeros(n_blocks, dtype=torch.float64), "n": 0}

class HookProc:
    def __init__(self, orig, layer_idx):
        self.orig = orig; self.layer_idx = layer_idx
    def __call__(self, attn_self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        B = hidden_states.shape[0]; img_n = hidden_states.shape[1]
        heads = attn_self.heads
        hd = attn_self.to_q.out_features // heads
        q = attn_self.to_q(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        k = attn_self.to_k(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        if getattr(attn_self, "norm_q", None) is not None:
            q = attn_self.norm_q(q); k = attn_self.norm_k(k)
        if encoder_hidden_states is not None:
            en = encoder_hidden_states.shape[1]
            eq = attn_self.add_q_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            ek = attn_self.add_k_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            if getattr(attn_self, "norm_added_q", None) is not None:
                eq = attn_self.norm_added_q(eq); ek = attn_self.norm_added_k(ek)
            q = torch.cat([q, eq], dim=2); k = torch.cat([k, ek], dim=2)
        probs = torch.softmax(q @ k.transpose(-1, -2) * (hd ** -0.5), dim=-1)
        with torch.no_grad():
            Aa = torch.tensor(A_idx, device=probs.device); Ba = torch.tensor(B_idx, device=probs.device)
            acc["AB"][self.layer_idx] += float(probs[:, :, Aa][:, :, :, Ba].mean().item())
            acc["BA"][self.layer_idx] += float(probs[:, :, Ba][:, :, :, Aa].mean().item())
        acc["n"] += 1
        return self.orig(attn_self, hidden_states, encoder_hidden_states, attention_mask, **kw)

for i in range(n_blocks):
    pipe.transformer.transformer_blocks[i].attn.processor = HookProc(
        type(pipe.transformer.transformer_blocks[i].attn.processor).__call__, i)
log(f"hook installed on {n_blocks} blocks")

# generation WITH hook
lat2 = initial.clone(); sched2 = copy.deepcopy(pipe.scheduler)
for i in range(STEPS):
    lat2 = sched2.step(cfg_full(0, lat2, timesteps[i]).to(lat2.dtype), timesteps[i],
                       lat2, return_dict=False)[0].to(initial.dtype)
log(f"hooked generation done, steps={acc['n']}")

ab = (acc["AB"] / max(acc["n"], 1)).numpy(); ba = (acc["BA"] / max(acc["n"], 1)).numpy()
avg = (ab + ba) / 2
order = np.argsort(-avg)
print("=== per-layer mean cross-entity attention (A<->B), 青花瓷 vs maiolica ===")
print(f"{'rank':>4} {'layer':>6} {'A->B':>8} {'B->A':>8} {'avg':>8}")
for rk, i in enumerate(order, 1):
    print(f"{rk:>4} {i:>6} {ab[i]:>6.4f} {ba[i]:>6.4f} {avg[i]:>6.4f}")
top = order[:14].tolist()
print(f"\nCRITICAL_LAYERS (top14) = {top}")
(OUT / "layer_interference.json").write_text(json.dumps(
    {"AB": ab.tolist(), "BA": ba.tolist(), "avg": avg.tolist(),
     "top_layers": top, "steps": acc["n"], "seed": SEED, "size": SIZE, "model": M}, indent=1),
    encoding="utf-8")
print(json.dumps({"event": "complete", "top": top, "steps": acc["n"]}), flush=True)
