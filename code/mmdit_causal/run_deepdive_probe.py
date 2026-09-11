"""Deep-dive internal probe at 1024 using the 100-pair prompts.

Measures, at each active (layer, step), the effect of changing routing vs content
on the written attention output: d_route = ||O_mixed - O_wfix||, d_value = ||O_mixed - O_vfix||.
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
    ap.add_argument("--out", type=Path, default=Path("/science/wx/pry/EXP1/experiment/deepdive_probe"))
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
                dw = float(np.linalg.norm(m - w))
                dv = float(np.linalg.norm(m - vv))
                rows.append({
                    "layer": rec["layer"], "step": rec["step"],
                    "d_route": dw, "d_value": dv,
                    "route_share": dw / (dw + dv) if (dw + dv) > 0 else None,
                    "value_share": dv / (dw + dv) if (dw + dv) > 0 else None,
                })
        act = [r for r in rows if r["step"] in args.steps_active and r["layer"] in args.layers]
        s = {
            "seed": seed, "n_active": len(act),
            "mean_d_route": float(np.mean([r["d_route"] for r in act])) if act else None,
            "mean_d_value": float(np.mean([r["d_value"] for r in act])) if act else None,
            "route_share": float(np.mean([r["route_share"] for r in act])) if act else None,
            "value_share": float(np.mean([r["value_share"] for r in act])) if act else None,
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
