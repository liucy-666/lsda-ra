from __future__ import annotations

import argparse
import json
import math
import time

import torch
from diffusers import StableDiffusion3Pipeline

from phase28_trainfree import AB_PROMPT, MODEL_DIR, ROOT, make_latent, save_png
from phase212_regional_score import (
    encode_conditions,
    prepare_schedule,
    soft_box_mask,
    transformer_pair,
)


OUT = ROOT / "phase213_multidiffusion_crop"

# Local canvases include context around each object and overlap slightly near the contact boundary.
PLATE_CROP = (0.00, 0.00, 0.625, 1.00)
VASE_CROP = (0.50, 0.09375, 1.00, 1.00)


def catalog():
    return [
        {
            "name": "mdcrop_w3_all",
            "region_weight": 3.0,
            "active_start": 0,
            "active_end": 28,
            "feather": 0.05,
            "plate_crop": PLATE_CROP,
            "vase_crop": VASE_CROP,
            "description": (
                "Faithful MultiDiffusion-style local-canvas score fusion: global A+B plus A-only plate crop "
                "and B-only vase crop at every denoising step"
            ),
        }
    ]


def crop_bounds(box, height, width, patch_size):
    x0, y0, x1, y1 = box
    left = max(0, int(math.floor(x0 * width / patch_size)) * patch_size)
    top = max(0, int(math.floor(y0 * height / patch_size)) * patch_size)
    right = min(width, int(math.ceil(x1 * width / patch_size)) * patch_size)
    bottom = min(height, int(math.ceil(y1 * height / patch_size)) * patch_size)
    if right <= left or bottom <= top:
        raise ValueError(f"Empty crop for {box}: {(left, top, right, bottom)}")
    return left, top, right, bottom


def region_cfg_score(pipe, crop, timestep, negative, positive, negative_pooled, positive_pooled, scale):
    pair_embeds = torch.cat([negative, positive], dim=0)
    pair_pooled = torch.cat([negative_pooled, positive_pooled], dim=0)
    uncond, cond = transformer_pair(pipe, crop, timestep, pair_embeds, pair_pooled).chunk(2)
    return uncond + scale * (cond - uncond)


def denoise(pipe, latent, encoded, row, steps, guidance_scale):
    latents = latent.clone()
    timesteps, mu = prepare_schedule(pipe, latents, steps)
    patch_size = int(pipe.transformer.config.patch_size)
    full_h, full_w = latents.shape[-2:]
    plate_bounds = crop_bounds(row["plate_crop"], full_h, full_w, patch_size)
    vase_bounds = crop_bounds(row["vase_crop"], full_h, full_w, patch_size)
    plate_mask = soft_box_mask(
        row["plate_crop"], full_h, full_w, row["feather"], latents.device
    )
    vase_mask = soft_box_mask(
        row["vase_crop"], full_h, full_w, row["feather"], latents.device
    )

    global_embeds = torch.cat([encoded["negative"], encoded["positive"][:1]], dim=0)
    global_pooled = torch.cat(
        [encoded["negative_pooled"], encoded["positive_pooled"][:1]], dim=0
    )
    weight = float(row["region_weight"])

    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            uncond, global_cond = transformer_pair(
                pipe, latents, timestep, global_embeds, global_pooled
            ).chunk(2)
            global_score = uncond + guidance_scale * (global_cond - uncond)

            if row["active_start"] <= index < min(row["active_end"], steps):
                numerator = global_score.float()
                denominator = torch.ones(
                    (1, 1, full_h, full_w), device=latents.device, dtype=torch.float32
                )
                for bounds, mask, positive, positive_pooled in (
                    (
                        plate_bounds,
                        plate_mask,
                        encoded["positive"][1:2],
                        encoded["positive_pooled"][1:2],
                    ),
                    (
                        vase_bounds,
                        vase_mask,
                        encoded["positive"][2:3],
                        encoded["positive_pooled"][2:3],
                    ),
                ):
                    left, top, right, bottom = bounds
                    crop = latents[:, :, top:bottom, left:right]
                    local_score = region_cfg_score(
                        pipe,
                        crop,
                        timestep,
                        encoded["negative"],
                        positive,
                        encoded["negative_pooled"],
                        positive_pooled,
                        guidance_scale,
                    )
                    local_mask = mask[:, :, top:bottom, left:right]
                    numerator[:, :, top:bottom, left:right] += (
                        weight * local_mask * local_score.float()
                    )
                    denominator[:, :, top:bottom, left:right] += weight * local_mask
                noise_pred = (numerator / denominator).to(global_score.dtype)
            else:
                noise_pred = global_score

            previous_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            if latents.dtype != previous_dtype:
                latents = latents.to(previous_dtype)

        scaled = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    diagnostics = {
        "mu": mu,
        "timesteps": len(timesteps),
        "patch_size": patch_size,
        "plate_bounds_latent": plate_bounds,
        "vase_bounds_latent": vase_bounds,
        "plate_crop_shape": (plate_bounds[3] - plate_bounds[1], plate_bounds[2] - plate_bounds[0]),
        "vase_crop_shape": (vase_bounds[3] - vase_bounds[1], vase_bounds[2] - vase_bounds[0]),
        "interior_own_fraction": weight / (1.0 + weight),
    }
    return image, diagnostics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = catalog()
    if args.names:
        wanted = set(args.names)
        rows = [row for row in rows if row["name"] in wanted]
        missing = wanted - {row["name"] for row in rows}
        if missing:
            raise ValueError(f"Unknown conditions: {sorted(missing)}")

    if args.smoke:
        steps, height, width, max_sequence_length = 2, 512, 512, 128
        run_root = OUT / "smoke"
        for row in rows:
            row["active_end"] = steps
    else:
        steps, height, width, max_sequence_length = 28, 1024, 1024, 256
        run_root = OUT / "formal"
    for sub in ("combined", "analysis", "logs"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    (run_root / "catalog.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    encoded = encode_conditions(pipe, max_sequence_length)

    for seed in args.seeds:
        latent, latent_sha = make_latent(pipe, seed, 1, height, width)
        for row in rows:
            path = run_root / "combined" / f"seed{seed:06d}__{row['name']}.png"
            if path.exists() and not args.overwrite:
                print(json.dumps({"event": "skipped", "seed": seed, "name": row["name"]}), flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            image, diagnostics = denoise(pipe, latent, encoded, row, steps, guidance_scale=4.5)
            config = {
                **row,
                **diagnostics,
                "source": "MultiDiffusion local-canvas least-squares fusion adapted to SD3.5",
                "same_current_ab_latent_crops": True,
                "no_donor_image_or_trajectory": True,
                "guidance_scale": 4.5,
                "steps": steps,
                "height": height,
                "width": width,
            }
            save_png(image, path, seed, AB_PROMPT, latent_sha, config)
            print(
                json.dumps(
                    {
                        "event": "completed",
                        "seed": seed,
                        "name": row["name"],
                        "seconds": time.time() - started,
                        "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "diagnostics": diagnostics,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
