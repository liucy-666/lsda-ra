"""A1-1: find SD3.5 layers critical for cultural attribute binding (cross-entity attention).

Generate one native SS (Chinese blue-white vase left, Italian maiolica right), hooking
every transformer block's joint attention to recompute attention weights and accumulate
cross-entity attention: image-token A(left) -> image-token B(right) and B->A, per layer,
averaged over all denoising steps. Layers with high cross-entity attention are where
attributes leak between instances = candidate critical layers (DreamRenderer-style).
Run on A100 GPU (SD3.5 + sam-vit-base + code/lsda).
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/science/wx/pry/MMDIT/code/lsda")
import numpy as np
import torch
import torch.nn.functional as F
import lsda_pipeline as L
from runtime import prepare_schedule, transformer_pair, region_cfg_score, encode_prompts
from diffusers import StableDiffusion3Pipeline
from transformers import SamModel, SamProcessor

M = "/science/wx/pry/models/stable-diffusion-3.5-large"
SAM = "/science/wx/pry/models/sam-vit-base"
OUT = Path("/science/wx/pry/MMDIT/exp_a1_1")
SEED = 1011
SIZE = 384  # small: keep SD3.5-Large peak well under VRAM; analysis only needs attention structure
STEPS = 28
CFG_ON = False  # single-batch forward (no classifier-free guidance) -> halves attention memory
GUIDANCE = 4.5
SS = ("Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
      "an Italian maiolica vase on the right; both fully visible, separate, and similar in size.")
PA = "a Chinese blue-and-white porcelain vase"
PB = "an Italian maiolica vase"
OUT.mkdir(parents=True, exist_ok=True)


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
# NOTE: do NOT call enable_attention_slicing() — its slicing processor materializes
# the full attention matrix (the fixed ~79GB OOM source). Default SDPA is memory-efficient.
helpers = (prepare_schedule, transformer_pair, region_cfg_score)
patch = int(pipe.transformer.config.patch_size)
initial, sha = L.make_latent(pipe, SEED)
ts, mu = prepare_schedule(pipe, initial, STEPS)
enc = encode_prompts(pipe, (SS, PA, PB))
log("model ready")

# native SS decode for SAM
def cfg_full(idx, lat, t):
    e = torch.cat([enc["negative"], enc["positive"][idx:idx + 1]], dim=0)
    p = torch.cat([enc["negative_pooled"], enc["positive_pooled"][idx:idx + 1]], dim=0)
    u, c = transformer_pair(pipe, lat, t, e, p).chunk(2)
    if not CFG_ON:
        return c
    return u + GUIDANCE * (c - u)

lat = initial.clone()
sched = copy.deepcopy(pipe.scheduler)
states = []
for i in range(STEPS):
    lat = sched.step(cfg_full(0, lat, ts[i]).to(lat.dtype), ts[i], lat, return_dict=False)[0].to(initial.dtype)
    states.append(lat.clone())
ss = L.decode(pipe, states[-1]); ss.save(OUT / "ss.png"); log("native done")

# SAM A/B
sam_proc = SamProcessor.from_pretrained(SAM, local_files_only=True)
sam_model = SamModel.from_pretrained(SAM, local_files_only=True).to("cuda").eval()
cA = L.sam_candidates(sam_proc, sam_model, ss, L.expected_x("left", 0, 2))
cB = L.sam_candidates(sam_proc, sam_model, ss, L.expected_x("right", 1, 2))
chosen = L.choose_joint([cA, cB])
owners_img, _, _ = L.clean_and_partition(chosen)
del sam_model, sam_proc; torch.cuda.empty_cache()
log("sam done")

# map image-space owner masks to latent token grid (H/patch x W/patch)
H_lat, W_lat = states[-1].shape[-2], states[-1].shape[-1]
H_tok, W_tok = H_lat // patch, W_lat // patch
def owner_tokens(idx):
    m = torch.from_numpy(owners_img[idx].astype(np.float32))[None, None].to("cuda")
    m = F.interpolate(m, size=(H_lat, W_lat), mode="nearest")[0, 0]  # [H_lat, W_lat]
    m = m.view(H_tok, patch, W_tok, patch).mean(dim=(1, 3)) > 0.5  # [H_tok, W_tok]
    return torch.nonzero(m).cpu().numpy()  # (n, 2) rows/cols
rem = owner_tokens(0); rbm = owner_tokens(1)
A_idx = set(int(r * W_tok + c) for r, c in rem)
B_idx = set(int(r * W_tok + c) for r, c in rbm)
A_idx = sorted(A_idx); B_idx = sorted(B_idx)
log(f"tokens A={len(A_idx)} B={len(B_idx)}")

# hook: recompute attention weights, accumulate image-A <-> B cross-entity attention per layer
n_blocks = len(pipe.transformer.transformer_blocks)
acc = {"AB": torch.zeros(n_blocks, dtype=torch.float64),
       "BA": torch.zeros(n_blocks, dtype=torch.float64), "n": 0}

class HookProc:
    def __init__(self, orig, layer_idx):
        self.orig = orig
        self.layer_idx = layer_idx

    def __call__(self, attn_self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        B = hidden_states.shape[0]
        img_n = hidden_states.shape[1]
        heads = attn_self.heads
        hd = attn_self.to_q.out_features // heads
        q = attn_self.to_q(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        k = attn_self.to_k(hidden_states).view(B, img_n, heads, hd).transpose(1, 2)
        if attn_self.norm_q is not None:
            q = attn_self.norm_q(q); k = attn_self.norm_k(k)
        if encoder_hidden_states is not None:
            en = encoder_hidden_states.shape[1]
            eq = attn_self.add_q_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            ek = attn_self.add_k_proj(encoder_hidden_states).view(B, en, heads, hd).transpose(1, 2)
            if attn_self.norm_added_q is not None:
                eq = attn_self.norm_added_q(eq); ek = attn_self.norm_added_k(ek)
            q = torch.cat([q, eq], dim=2); k = torch.cat([k, ek], dim=2)
        probs = torch.softmax(q @ k.transpose(-1, -2) * (hd ** -0.5), dim=-1)  # [B,heads,seq,seq]
        with torch.no_grad():
            A_arr = torch.tensor(A_idx, device=probs.device)
            B_arr = torch.tensor(B_idx, device=probs.device)
            a2b = probs[:, :, A_arr][:, :, :, B_arr].mean()
            b2a = probs[:, :, B_arr][:, :, :, A_arr].mean()
        acc["AB"][self.layer_idx] += float(a2b.item())
        acc["BA"][self.layer_idx] += float(b2a.item())
        acc["n"] += 1
        return self.orig(attn_self, hidden_states, encoder_hidden_states, attention_mask, **kw)


for i in range(n_blocks):
    pipe.transformer.transformer_blocks[i].attn.processor = HookProc(
        type(pipe.transformer.transformer_blocks[i].attn.processor).__call__, i)

# re-run generation WITH hook (fresh from initial) to accumulate across steps
lat2 = initial.clone(); sched2 = copy.deepcopy(pipe.scheduler)
for i in range(STEPS):
    lat2 = sched2.step(cfg_full(0, lat2, ts[i]).to(lat2.dtype), ts[i], lat2, return_dict=False)[0].to(initial.dtype)
log(f"hooked generation done, steps={acc['n']}")

ab = (acc["AB"] / max(acc["n"], 1)).numpy()
ba = (acc["BA"] / max(acc["n"], 1)).numpy()
avg = (ab + ba) / 2
order = np.argsort(-avg)
print("=== per-layer mean cross-entity attention (A<->B) ===")
print(f"{'rank':>4} {'layer':>6} {'A->B':>8} {'B->A':>8} {'avg':>8}")
for rank, i in enumerate(order, 1):
    print(f"{rank:>4} {i:>6} {ab[i]:>8.4f} {ba[i]:>8.4f} {avg[i]:>8.4f}")
top = order[: min(14, n_blocks)]
print(f"\nCRITICAL_LAYERS={top.tolist()}")
(OUT / "layer_interference.json").write_text(json.dumps({
    "AB": ab.tolist(), "BA": ba.tolist(), "avg": avg.tolist(),
    "top_layers": top.tolist(), "steps": acc["n"], "seed": SEED, "size": SIZE,
}, indent=1), encoding="utf-8")
print(json.dumps({"event": "complete", "top": top.tolist(), "steps": acc["n"]}), flush=True)
