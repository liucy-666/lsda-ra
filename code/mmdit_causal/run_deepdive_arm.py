"""Deep-dive causal arms at 1024 using the 100-pair prompts.

Arms: baseline / w_fix / v_fix / both_fix / h_fix.
Donor = B-only branch at the same position (right), same forward.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, Controller, install, make_latent
from pairs100 import b_donor_prompt, ss_prompt

ARMS = ["baseline", "w_fix", "v_fix", "both_fix", "h_fix"]


def run_arm(pipe, controller, pair_idx, seed, steps, size, guidance, out_dir, arm):
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"pair_{pair_idx:03d}_seed{seed}_{arm}.png"
    donor = out_dir / f"pair_{pair_idx:03d}_seed{seed}_{arm}_donorB.png"
    if target.exists() and donor.exists():
        return {"event": "skipped", "path": str(target)}
    latent, sha = make_latent(pipe, seed, 2, size, size)
    controller.reset()
    prompts = [ss_prompt(pair_idx), b_donor_prompt(pair_idx)]
    t0 = time.time()
    with torch.inference_mode():
        images = pipe(
            prompt=prompts, prompt_2=prompts, prompt_3=prompts,
            negative_prompt=["", ""], negative_prompt_2=["", ""], negative_prompt_3=["", ""],
            num_inference_steps=steps, guidance_scale=guidance,
            height=size, width=size, latents=latent, output_type="pil",
        ).images
    images[0].save(target)
    images[1].save(donor)
    meta = {
        "pair": f"pair_{pair_idx:03d}", "pair_idx": pair_idx, "seed": seed, "arm": arm,
        "layers": list(controller.layers), "steps_active": list(controller.steps),
        "size": size, "steps": steps, "guidance": guidance, "latent_sha256": sha,
        "seconds": round(time.time() - t0, 1),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    (out_dir / f"pair_{pair_idx:03d}_seed{seed}_{arm}.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )
    return {"event": "completed", **meta, "path": str(target)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["v_fix", "h_fix"])
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(38)))
    ap.add_argument("--steps-active", type=int, nargs="+", default=list(range(28)))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/data/deepdive1024/arms"))
    args = ap.parse_args()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for arm in args.arms:
        controller = Controller(grid=args.size // 16, layers=args.layers, steps=args.steps_active, arm=arm)
        install(pipe, controller)
        for seed in args.seeds:
            res = run_arm(pipe, controller, args.pair, seed, args.steps, args.size, args.guidance, args.out, arm)
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
