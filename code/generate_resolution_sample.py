"""Generate one SD3.5 Large image at a requested square resolution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator("cuda").manual_seed(args.seed)
    started = time.time()
    image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=4.5,
        height=args.size,
        width=args.size,
        max_sequence_length=256,
        generator=generator,
    ).images[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(json.dumps({
        "output": str(args.output), "size": args.size, "seed": args.seed,
        "seconds": round(time.time() - started, 3),
        "bytes": args.output.stat().st_size,
    }))


if __name__ == "__main__":
    main()
