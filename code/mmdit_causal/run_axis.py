"""Culture-axis recorder (Phase 5).

For one pair:
  1. Build the A->B culture axis per layer from standalone A-right / B-right controls
     (ROI = right-object token region, capture = ROI-mean of h_in).
  2. Run each mixed seed with capture and project h_in / attn_gated / f_gated onto the axis.

Output: per (seed, layer, step) projections p_in / p_attn / p_ffn.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, Controller, install, make_latent
from pairs100 import PAIRS

WINDOW = (24, 25, 26, 27)
CAPTURE_NAMES = ("h_in", "attn_gated", "f_gated")


def run_capture(pipe, controller, prompts, seed, size, steps, guidance, branch):
    controller.reset()
    controller.capture = {}
    controller.capture_branch = branch
    latent, _ = make_latent(pipe, seed, len(prompts), size, size)
    with torch.inference_mode():
        pipe(
            prompt=prompts, prompt_2=prompts, prompt_3=prompts,
            negative_prompt=[""] * len(prompts), negative_prompt_2=[""] * len(prompts),
            negative_prompt_3=[""] * len(prompts),
            num_inference_steps=steps, guidance_scale=guidance,
            height=size, width=size, latents=latent, output_type="pil",
        )
    return controller.capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True, help="1-based pair index (e.g. 3 for pair_003)")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--cohort", nargs="+", default=None, help="failed/normal per seed (optional)")
    ap.add_argument("--axis-seeds", type=int, nargs="+", default=[1011, 2021])
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(38)))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/experiment/axis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"axis_pair{args.pair:03d}.json"
    if out_path.exists():
        print(json.dumps({"event": "skip", "path": str(out_path)}), flush=True)
        return

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    grid = args.size // 16

    controller = Controller(grid=grid, layers=args.layers, steps=list(range(args.steps)), arm="baseline")
    controller.capture_enabled = True
    controller.capture_roi_only = True
    install(pipe, controller, patch_attention=False)

    ent = PAIRS[args.pair]
    a_right = f"Neutral studio background: {ent['A']} on the right; fully visible."
    b_right = f"Neutral studio background: {ent['B']} on the right; fully visible."

    # ---- 1) axis from standalone controls
    mu_a, mu_b = {}, {}
    for seed in args.axis_seeds:
        cap_a = run_capture(pipe, controller, [a_right], seed, args.size, args.steps, args.guidance, branch=1)
        cap_b = run_capture(pipe, controller, [b_right], seed, args.size, args.steps, args.guidance, branch=1)
        for layer in args.layers:
            for step in WINDOW:
                if step not in cap_a.get(layer, {}).get("h_in", {}):
                    continue
                va = cap_a[layer]["h_in"][step]
                vb = cap_b[layer]["h_in"][step]
                mu_a.setdefault((layer, step), []).append(va)
                mu_b.setdefault((layer, step), []).append(vb)
    axis = {}
    axis_norm = {}
    for key in mu_a:
        va = np.mean(mu_a[key], axis=0)
        vb = np.mean(mu_b[key], axis=0)
        d = va - vb
        n = float(np.linalg.norm(d))
        axis[key] = d / n if n > 0 else d
        axis_norm[key] = n

    # ---- 2) project mixed seeds
    projections = {}
    cohorts = args.cohort or ["unknown"] * len(args.seeds)
    for seed, cohort in zip(args.seeds, cohorts):
        cap = run_capture(pipe, controller, [ent["SS"], b_right], seed, args.size, args.steps, args.guidance, branch=2)
        rec = {"cohort": cohort, "p_in": {}, "p_attn": {}, "p_ffn": {}, "norm_in": {}, "norm_attn": {}}
        for layer in args.layers:
            for step in range(args.steps):
                h = cap.get(layer, {}).get("h_in", {}).get(step)
                ag = cap.get(layer, {}).get("attn_gated", {}).get(step)
                fg = cap.get(layer, {}).get("f_gated", {}).get(step)
                key = (layer, step)
                if key in axis and h is not None:
                    rec["p_in"].setdefault(layer, {})[step] = float(np.dot(h, axis[key]))
                    rec["norm_in"].setdefault(layer, {})[step] = float(np.linalg.norm(h))
                if key in axis and ag is not None:
                    rec["p_attn"].setdefault(layer, {})[step] = float(np.dot(ag, axis[key]))
                    rec["norm_attn"].setdefault(layer, {})[step] = float(np.linalg.norm(ag))
                if key in axis and fg is not None:
                    rec["p_ffn"].setdefault(layer, {})[step] = float(np.dot(fg, axis[key]))
        projections[str(seed)] = rec
        print(json.dumps({"event": "seed_done", "seed": seed, "cohort": cohort}), flush=True)

    out = {
        "pair": f"pair_{args.pair:03d}", "pair_idx": args.pair,
        "axis_seeds": args.axis_seeds, "window": list(WINDOW),
        "axis_norm": {f"{l}|{s}": v for (l, s), v in axis_norm.items()},
        "projections": projections,
    }
    (args.out / f"axis_pair{args.pair:03d}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"event": "axis_done", "pair": args.pair, "n_seeds": len(args.seeds)}), flush=True)


if __name__ == "__main__":
    main()
