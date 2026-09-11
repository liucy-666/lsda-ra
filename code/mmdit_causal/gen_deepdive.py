"""Phase 3: targeted regeneration at 1024 for the deep-dive pairs.

Generates, for each target pair:
  - native SS for all seeds in the job list (failed + normal)
  - standalone A and B for the failed seeds (inclusion criterion)

Jobs come from deepdive_jobs.json:
  {"pairs": {"pair_003": {"failed": [...], "normal": [...]}, ...}}
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
from diffusers import StableDiffusion3Pipeline

from mmdit_lib import MODEL_DIR, make_latent
from pairs100 import a_only_prompt, b_only_prompt, ss_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/data/deepdive1024/e0"))
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--only", choices=["ss", "standalone", "all"], default="all")
    args = ap.parse_args()

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["pairs"]
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for pair_name, spec in sorted(jobs.items()):
        pair_idx = int(pair_name.split("_")[1])
        failed = spec["failed"]
        normal = spec.get("normal", [])

        tasks = []
        if args.only in ("ss", "all"):
            for seed in failed + normal:
                tasks.append(("SS", seed, ss_prompt(pair_idx)))
        if args.only in ("standalone", "all"):
            for seed in failed:
                tasks.append(("A", seed, a_only_prompt(pair_idx)))
                tasks.append(("B", seed, b_only_prompt(pair_idx)))

        for kind, seed, prompt in tasks:
            path = args.out / f"{pair_name}_seed{seed}_{kind}.png"
            if path.exists():
                print(json.dumps({"event": "skip", "path": str(path)}), flush=True)
                continue
            generator = torch.Generator("cuda").manual_seed(seed)
            t0 = time.time()
            img = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                height=args.size,
                width=args.size,
                generator=generator,
            ).images[0]
            img.save(path)
            meta = {
                "pair": pair_name, "pair_idx": pair_idx, "seed": seed, "kind": kind,
                "prompt": prompt, "size": args.size, "steps": args.steps,
                "guidance": args.guidance, "seconds": round(time.time() - t0, 1),
                "cohort": "failed" if seed in failed else "normal",
            }
            (args.out / f"{pair_name}_seed{seed}_{kind}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(json.dumps({"event": "done", "pair": pair_name, "seed": seed, "kind": kind,
                              "seconds": meta["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
