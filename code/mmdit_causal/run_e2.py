"""E2: 2x2 factorial causal test (baseline / w_fix / v_fix / both_fix).

Each arm runs one forward with prompts [mixed AB, B-only-at-right] (CFG -> batch 4):
  branch 2 = mixed AB (target), branch 3 = B-only (donor).
Patches are applied only at (layer, step), on the right-object token region.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, Controller, install, run_arm

ARMS = ["baseline", "w_fix", "v_fix", "both_fix"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[36])
    ap.add_argument("--steps-active", type=int, nargs="+", default=[26])
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/data/e2"))
    args = ap.parse_args()

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    grid = args.size // 16

    for arm in args.arms:
        controller = Controller(grid=grid, layers=args.layers, steps=args.steps_active, arm=arm)
        install(pipe, controller)
        for seed in args.seeds:
            res = run_arm(
                pipe, controller, args.pair, seed, args.steps, args.size,
                args.guidance, args.out, arm,
            )
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
