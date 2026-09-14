"""KA/ME census generation: standalone A/B + native SS for 100 cultural pairs x 3 seeds.

Output naming: pair{idx:03d}_seed{seed}_{A|B|SS}.png (+ .json sidecar), compatible with
build_e0_manifest.py.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from diffusers import StableDiffusion3Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt  # noqa: E402

MODEL = os.environ.get("CENSUS_MODEL", "/science/wx/pry/models/stable-diffusion-3.5-large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1011, 1012, 1013])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    builders = {"SS": ss_prompt, "A": a_only_prompt, "B": b_only_prompt}
    for idx in args.pairs:
        for seed in args.seeds:
            for kind, builder in builders.items():
                path = args.out / f"pair{idx:03d}_seed{seed}_{kind}.png"
                if path.exists():
                    continue
                prompt = builder(idx)
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
                        {
                            "pair": idx, "seed": seed, "kind": kind, "prompt": prompt,
                            "size": args.size, "steps": args.steps, "guidance": args.guidance,
                            "seconds": round(time.time() - t0, 1),
                        },
                        ensure_ascii=False, indent=1,
                    ),
                    encoding="utf-8",
                )
                print(json.dumps({"done": path.name, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
