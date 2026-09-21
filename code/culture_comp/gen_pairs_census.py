"""Manifest-driven census generator: standalone A/B + native SS for arbitrary pairs.

Pairs JSON (list):
  [{"idx": 1, "A": {"short": "..."}, "B": {"short": "..."}, "ss_prompt": "..."}, ...]

Output naming: pair{idx:03d}_seed{seed}_{A|B|SS}.png (+ .json sidecar),
compatible with code/lsda/run_census_lsda.py --images-dir.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from diffusers import StableDiffusion3Pipeline

MODEL = os.environ.get("CENSUS_MODEL", "/science/wx/pry/models/stable-diffusion-3.5-large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--kinds", default="A,B,SS", help="comma list of A,B,SS")
    args = ap.parse_args()

    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for rec in pairs:
        idx = rec["idx"]
        prompts = {"A": rec["A"]["short"], "B": rec["B"]["short"], "SS": rec["ss_prompt"]}
        kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
        for seed in args.seeds:
            for kind in kinds:
                prompt = prompts[kind]
                path = args.out / f"pair{idx:03d}_seed{seed}_{kind}.png"
                if path.exists():
                    continue
                t0 = time.time()
                generator = torch.Generator("cuda").manual_seed(seed)
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    height=args.size,
                    width=args.size,
                    generator=generator,
                ).images[0]
                image.save(path)
                (args.out / f"pair{idx:03d}_seed{seed}_{kind}.json").write_text(
                    json.dumps(
                        {"pair": idx, "seed": seed, "kind": kind, "prompt": prompt,
                         "size": args.size, "steps": args.steps, "guidance": args.guidance,
                         "seconds": round(time.time() - t0, 1)},
                        ensure_ascii=False, indent=1,
                    ),
                    encoding="utf-8",
                )
                print(json.dumps({"done": path.name, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
