"""Deep-dive internal probe at 1024 with DIRECTION-AWARE metrics.

At each active (layer, step), for the target region:
  O_mixed = W_mixed . V_mixed
  O_wfix  = W_donor . V_mixed
  O_vfix  = W_mixed . V_donor
  O_donor = W_donor . V_donor

Direction-aware metrics (does the swap move TOWARD the donor?):
  gap        = ||O_mixed - O_donor||
  recovery_W = 1 - ||O_wfix - O_donor|| / gap
  recovery_V = 1 - ||O_vfix - O_donor|| / gap
  proj_W     = <O_wfix - O_mixed, O_donor - O_mixed> / ||O_donor - O_mixed||^2
  proj_V     = <O_vfix - O_mixed, O_donor - O_mixed> / ||O_donor - O_mixed||^2

Also stores the four raw mean vectors for offline re-analysis.
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
from pairs100 import b_donor_prompt, ss_prompt
from run_internal import InternalProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[33, 34, 35, 36])
    ap.add_argument("--steps-active", type=int, nargs="+", default=[24, 25, 26, 27])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--save-raw", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/experiment/deepdive_probe_v2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
    summary = {}
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
                gap = float(np.linalg.norm(m - d))
                tgt = d - m
                tgt2 = float(np.dot(tgt, tgt))
                row = {
                    "layer": rec["layer"], "step": rec["step"],
                    "d_route": float(np.linalg.norm(m - w)),
                    "d_value": float(np.linalg.norm(m - vv)),
                    "gap": gap,
                    "recovery_W": 1 - float(np.linalg.norm(w - d)) / gap if gap > 0 else None,
                    "recovery_V": 1 - float(np.linalg.norm(vv - d)) / gap if gap > 0 else None,
                    "proj_W": float(np.dot(w - m, tgt)) / tgt2 if tgt2 > 0 else None,
                    "proj_V": float(np.dot(vv - m, tgt)) / tgt2 if tgt2 > 0 else None,
                }
                if args.save_raw:
                    row["vec_mixed"] = m.tolist()
                    row["vec_wfix"] = w.tolist()
                    row["vec_vfix"] = vv.tolist()
                    row["vec_donor"] = d.tolist()
                rows.append(row)
        act = [r for r in rows if r["step"] in args.steps_active and r["layer"] in args.layers]
        s = {
            "seed": seed, "n_active": len(act),
            "mean_gap": float(np.mean([r["gap"] for r in act])),
            "mean_recovery_W": float(np.mean([r["recovery_W"] for r in act if r["recovery_W"] is not None])),
            "mean_recovery_V": float(np.mean([r["recovery_V"] for r in act if r["recovery_V"] is not None])),
            "mean_proj_W": float(np.mean([r["proj_W"] for r in act if r["proj_W"] is not None])),
            "mean_proj_V": float(np.mean([r["proj_V"] for r in act if r["proj_V"] is not None])),
        }
        summary[str(seed)] = s
        (args.out / f"probe_pair{args.pair:03d}_seed{seed}.json").write_text(
            json.dumps({"pair": args.pair, "summary": s, "rows": rows}, indent=1), encoding="utf-8"
        )
        print(json.dumps({"event": "seed_done", **s}), flush=True)

    (args.out / f"probe_pair{args.pair:03d}_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    print(json.dumps({"event": "probe_done", "pair": args.pair, "n_seeds": len(args.seeds)}), flush=True)


if __name__ == "__main__":
    main()
