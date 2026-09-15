"""Generate LL images (global long prompt) using the same native-SS code path as EXP_3.

Jobs: [{run_id, pair, seed, prompt}]. Output: <out>/<run_id>.png (+ sidecar json).
Resumable (skips existing png).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

CODE = Path("/science/wx/pry/MMDIT/code/lsda")
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "helpers"))

import lsda_pipeline_v14_rect as L  # noqa: E402
from phase212_regional_score import prepare_schedule, transformer_pair  # noqa: E402
from phase221_group_overlap_arbitration import encode_prompts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    args = ap.parse_args()

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (prepare_schedule, transformer_pair, None)

    done = 0
    for job in jobs:
        png = args.out / f"{job['run_id']}.png"
        if png.exists():
            continue
        started = time.time()
        try:
            initial, sha = L.make_latent(pipe, int(job["seed"]))
            encoded = encode_prompts(pipe, (job["prompt"],))
            states, _, _ = L.native_ss_trajectory(pipe, initial, encoded, helpers)
            img = L.decode(pipe, states[-1])
            img.save(png)
            (args.out / f"{job['run_id']}.json").write_text(json.dumps({
                "pair": job["pair"], "seed": job["seed"], "kind": "LL",
                "prompt": job["prompt"], "size": 1024, "steps": 28, "guidance": 4.5,
                "latent_sha": sha, "seconds": round(time.time() - started, 1),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            done += 1
            print(json.dumps({"done": job["run_id"], "n": done, "sec": round(time.time() - started, 1)}), flush=True)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"error": job["run_id"], "msg": f"{type(exc).__name__}: {exc}"}), flush=True)
    print(json.dumps({"ALL_DONE": done}), flush=True)


if __name__ == "__main__":
    main()
