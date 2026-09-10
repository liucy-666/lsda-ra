"""E0: generate standalone A/B and mixed SS images for inclusion screening."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, a_only_prompt, b_only_prompt, ss_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/data/e0"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    conditions = [("A", a_only_prompt), ("B", b_only_prompt), ("SS", ss_prompt)]
    for seed in args.seeds:
        for cond, fn in conditions:
            path = args.out / f"pair{args.pair}_seed{seed}_{cond}.png"
            if path.exists():
                print(json.dumps({"event": "skip", "path": str(path)}), flush=True)
                continue
            generator = torch.Generator("cuda").manual_seed(seed)
            img = pipe(
                prompt=fn(args.pair),
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.size,
                width=args.size,
                generator=generator,
            ).images[0]
            img.save(path)
            print(json.dumps({"event": "done", "pair": args.pair, "seed": seed, "cond": cond}), flush=True)


if __name__ == "__main__":
    main()
