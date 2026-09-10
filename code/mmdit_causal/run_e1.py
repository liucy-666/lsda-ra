"""E1: reproduce prior findings on SD3.5 (no intervention).

E1.1  block identity decomposition at (layer, step): verify
      attn_gated = gate_msa * attn_raw, h_attn = h_in + attn_gated,
      f_gated = gate_mlp * ff(...), h_out = h_attn + f_gated.
E1.2  per-layer cross-entity attention (DreamRenderer-style): mean attention
      from left-object image tokens to right-object image tokens (and vice versa),
      using the mixed AB conditional branch (batch index 2).
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import numpy as np
import torch

from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, Controller, install, make_latent, ss_prompt, b_donor_prompt


class ProbeWrap:
    """Measure cross-entity attention, then delegate to the inner processor."""

    def __init__(self, inner, layer, acc, left_idx, right_idx):
        self.inner = inner
        self.layer = layer
        self.acc = acc
        self.left = left_idx
        self.right = right_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        with torch.no_grad():
            B = hidden_states.shape[0]
            img_n = hidden_states.shape[1]
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            hd = k.shape[-1] // attn.heads
            q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
            k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)
            if encoder_hidden_states is not None:
                cq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
                ck = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
                if attn.norm_added_q is not None:
                    cq = attn.norm_added_q(cq)
                if attn.norm_added_k is not None:
                    ck = attn.norm_added_k(ck)
                q = torch.cat([q, cq], 2)
                k = torch.cat([k, ck], 2)
            probs = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (hd ** -0.5), dim=-1)
            li = self.left.to(probs.device)
            ri = self.right.to(probs.device)
            ab = probs[2][:, li][:, :, ri].mean().item()
            ba = probs[2][:, ri][:, :, li].mean().item()
            self.acc["ab"][self.layer] += ab
            self.acc["ba"][self.layer] += ba
            self.acc["n"] += 1
        return self.inner(attn, hidden_states, encoder_hidden_states, attention_mask, *args, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--step", type=int, default=26)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/experiment/e1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    grid = args.size // 16
    n_blocks = len(pipe.transformer.transformer_blocks)

    # --- E1.1 controller capture on the active layer
    controller = Controller(grid=grid, layers=[args.layer], steps=[args.step], arm="baseline")
    controller.capture_enabled = True
    install(pipe, controller)

    # --- E1.2 probe every layer
    left_idx = torch.tensor([y * grid + x for y in range(grid) for x in range(grid // 2)], dtype=torch.long)
    right_idx = torch.tensor([y * grid + x for y in range(grid) for x in range(grid // 2, grid)], dtype=torch.long)
    acc = {"ab": torch.zeros(n_blocks), "ba": torch.zeros(n_blocks), "n": 0}
    for i in range(n_blocks):
        orig = pipe.transformer.transformer_blocks[i].attn.processor
        pipe.transformer.transformer_blocks[i].attn.set_processor(ProbeWrap(orig, i, acc, left_idx, right_idx))

    latent, latent_sha = make_latent(pipe, args.seed, 2, args.size, args.size)
    prompts = [ss_prompt(args.pair), b_donor_prompt(args.pair)]
    with torch.inference_mode():
        pipe(
            prompt=prompts, prompt_2=prompts, prompt_3=prompts,
            negative_prompt=["", ""], negative_prompt_2=["", ""], negative_prompt_3=["", ""],
            num_inference_steps=args.steps, guidance_scale=args.guidance,
            height=args.size, width=args.size, latents=latent, output_type="pil",
        )

    n = max(acc["n"], 1)
    ab = (acc["ab"] / n).numpy().tolist()
    ba = (acc["ba"] / n).numpy().tolist()
    avg = ((acc["ab"] + acc["ba"]) / (2 * n)).numpy().tolist()
    order = sorted(range(n_blocks), key=lambda i: -avg[i])

    # --- E1.1 consistency checks
    cap = controller.capture.get(args.layer, {})
    checks = {}
    if cap:
        for name in ("h_in", "attn_raw", "attn_gated", "h_attn", "f_raw", "f_gated", "h_out"):
            if name in cap:
                checks[name + "_norm"] = float(cap[name].norm())
        if "attn_gated" in cap and "gate_msa" in cap and "attn_raw" in cap:
            recon = cap["gate_msa"].unsqueeze(1) * cap["attn_raw"]
            checks["attn_gated_recon_rel_err"] = float((recon - cap["attn_gated"]).norm() / (cap["attn_gated"].norm() + 1e-9))
        if "h_out" in cap and "h_attn" in cap and "f_gated" in cap:
            recon = cap["h_attn"] + cap["f_gated"]
            checks["h_out_recon_rel_err"] = float((recon - cap["h_out"]).norm() / (cap["h_out"].norm() + 1e-9))
        if "f_gated" in cap and "gate_mlp" in cap and "f_raw" in cap:
            recon = cap["gate_mlp"].unsqueeze(1) * cap["f_raw"]
            checks["f_gated_recon_rel_err"] = float((recon - cap["f_gated"]).norm() / (cap["f_gated"].norm() + 1e-9))

    result = {
        "pair": args.pair, "seed": args.seed, "layer": args.layer, "step": args.step,
        "steps": args.steps, "size": args.size, "n_blocks": n_blocks, "probe_calls": acc["n"],
        "ab": ab, "ba": ba, "avg": avg, "top_layers": order[:12],
        "checks": checks,
    }
    (args.out / f"e1_pair{args.pair}_seed{args.seed}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"event": "e1_done", "top_layers": order[:12], "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
