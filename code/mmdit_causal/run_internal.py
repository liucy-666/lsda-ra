"""Internal 2x2 probe: at each active (layer, step), compute the four attention-output
variants for the target region in a SINGLE forward and measure which factor (routing W
or value V) moves the written content toward the donor (B-only) branch.

  O_mixed = W_mixed . V_mixed     (baseline)
  O_wfix  = W_donor . V_mixed     (fix routing only)
  O_vfix  = W_mixed . V_donor     (fix content only)
  O_donor = W_donor . V_donor     (fix both)

Recovery toward donor:
  r_w = 1 - ||O_wfix - O_donor|| / ||O_mixed - O_donor||
  r_v = 1 - ||O_vfix - O_donor|| / ||O_mixed - O_donor||
r_v > r_w  => content (value) dominates.
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


class InternalProbe:
    def __init__(self, controller, layer, grid):
        self.controller = controller
        self.layer = layer
        self.grid = grid
        self.records = []

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        B = hidden_states.shape[0]
        image_tokens = hidden_states.shape[1]
        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        hd = k.shape[-1] // attn.heads
        q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
        k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
        v = v.view(B, -1, attn.heads, hd).transpose(1, 2)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if encoder_hidden_states is not None:
            cq = attn.add_q_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            ck = attn.add_k_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            cv = attn.add_v_proj(encoder_hidden_states).view(B, -1, attn.heads, hd).transpose(1, 2)
            if attn.norm_added_q is not None:
                cq = attn.norm_added_q(cq)
            if attn.norm_added_k is not None:
                ck = attn.norm_added_k(ck)
            q = torch.cat([q, cq], 2)
            k = torch.cat([k, ck], 2)
            v = torch.cat([v, cv], 2)

        scale = hd ** -0.5
        w = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        states = torch.matmul(w, v)
        with torch.no_grad():
            w_mixed, w_donor = w[2], w[3]
            v_mixed, v_donor = v[2], v[3]
            idx = self.controller.target_indices(w_mixed.device)
            o_mixed = torch.matmul(w_mixed[:, idx, :], v_mixed)
            o_wfix = torch.matmul(w_donor[:, idx, :], v_mixed)
            o_vfix = torch.matmul(w_mixed[:, idx, :], v_donor)
            o_donor = torch.matmul(w_donor[:, idx, :], v_donor)
            self.records.append({
                "layer": self.layer,
                "step": self.controller.step_idx,
                "mixed": o_mixed.mean(dim=(0, 1)).float().cpu().numpy(),
                "wfix": o_wfix.mean(dim=(0, 1)).float().cpu().numpy(),
                "vfix": o_vfix.mean(dim=(0, 1)).float().cpu().numpy(),
                "donor": o_donor.mean(dim=(0, 1)).float().cpu().numpy(),
            })
        combined = states.transpose(1, 2).reshape(B, -1, attn.heads * hd).to(q.dtype)
        enc_out = None
        if encoder_hidden_states is not None:
            combined, enc_out = combined[:, :image_tokens], combined[:, image_tokens:]
            if not attn.context_pre_only:
                enc_out = attn.to_add_out(enc_out)
        combined = attn.to_out[1](attn.to_out[0](combined))
        return (combined, enc_out) if encoder_hidden_states is not None else combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[33, 34, 35, 36])
    ap.add_argument("--steps-active", type=int, nargs="+", default=[24, 25, 26, 27])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/experiment/e2_internal"))
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

    controller = Controller(grid=grid, layers=args.layers, steps=args.steps_active, arm="baseline")
    install(pipe, controller)
    probes = {}
    for layer in args.layers:
        probe = InternalProbe(controller, layer, grid)
        probes[layer] = probe
        pipe.transformer.transformer_blocks[layer].attn.set_processor(probe)

    prompts = [ss_prompt(args.pair), b_donor_prompt(args.pair)]
    all_summary = {}
    for seed in args.seeds:
        for probe in probes.values():
            probe.records = []
        controller.reset()
        latent, _ = make_latent(pipe, seed, 2, args.size, args.size)
        with torch.inference_mode():
            pipe(
                prompt=prompts, prompt_2=prompts, prompt_3=prompts,
                negative_prompt=["", ""], negative_prompt_2=["", ""], negative_prompt_3=["", ""],
                num_inference_steps=args.steps, guidance_scale=args.guidance,
                height=args.size, width=args.size, latents=latent, output_type="pil",
            )
        rows = []
        for layer, probe in probes.items():
            for rec in probe.records:
                m, w, vv, d = rec["mixed"], rec["wfix"], rec["vfix"], rec["donor"]
                active = rec["step"] in args.steps_active and rec["layer"] in args.layers
                dw = float(np.linalg.norm(m - w))
                dv = float(np.linalg.norm(m - vv))
                base = float(np.linalg.norm(m - d))
                rows.append({
                    "layer": rec["layer"], "step": rec["step"], "active": active,
                    "d_route": dw, "d_value": dv, "d_total": base,
                    "route_share": dw / (dw + dv) if (dw + dv) > 0 else None,
                    "value_share": dv / (dw + dv) if (dw + dv) > 0 else None,
                })
        act = [r for r in rows if r["active"]]
        summary = {
            "seed": seed, "n_active": len(act),
            "mean_d_route": float(np.mean([r["d_route"] for r in act])),
            "mean_d_value": float(np.mean([r["d_value"] for r in act])),
            "route_share": float(np.mean([r["route_share"] for r in act])),
            "value_share": float(np.mean([r["value_share"] for r in act])),
        }
        all_summary[str(seed)] = summary
        (args.out / f"internal_pair{args.pair}_seed{seed}.json").write_text(
            json.dumps({"pair": args.pair, "seed": seed, "summary": summary, "rows": rows}, indent=1), encoding="utf-8"
        )
        print(json.dumps({"event": "seed_done", **summary}), flush=True)

    (args.out / f"internal_pair{args.pair}_summary.json").write_text(
        json.dumps(all_summary, indent=1), encoding="utf-8"
    )
    print(json.dumps({"event": "internal_done", "n_seeds": len(args.seeds)}), flush=True)


if __name__ == "__main__":
    main()
