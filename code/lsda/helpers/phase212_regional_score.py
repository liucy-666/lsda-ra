from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)

from phase28_trainfree import AB_PROMPT, B_PROMPT, MODEL_DIR, ROOT, make_latent, save_png


OUT = ROOT / "phase212_regional_score"
A_PROMPT = (
    "A single Chinese blue-and-white porcelain plate with a luminous white glazed surface, "
    "decorated with cobalt-blue traditional Chinese landscape, pavilion, floral, and scroll motifs."
)

# These boxes describe where the two objects already occur in the unmodified seed-0 composition.
# They select score fields only; no A-only/B-only pixels, latents, activations, or trajectories are copied.
PLATE_BOX = (0.00, 0.02, 0.56, 0.99)
VASE_BOX = (0.56, 0.05, 0.99, 0.99)


def catalog():
    return [
        {
            "name": "regional_score_w3_all",
            "region_weight": 3.0,
            "active_start": 0,
            "active_end": 28,
            "feather": 0.04,
            "description": (
                "MultiDiffusion-style least-squares score fusion: 25% global A+B and 75% own-entity "
                "conditional score in each region, at all denoising steps"
            ),
        },
        {
            "name": "regional_score_w3_s08to19",
            "region_weight": 3.0,
            "active_start": 8,
            "active_end": 20,
            "feather": 0.04,
            "description": (
                "Geometry-preserving schedule selected from the all-step failure: global A+B for steps 0-7, "
                "75% own-entity regional score for steps 8-19, then global A+B refinement for steps 20-27"
            ),
        },
    ]


def smoothstep(x):
    x = x.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def soft_box_mask(box, height, width, feather, device):
    x0, y0, x1, y1 = box
    ys = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
    xs = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width

    def axis_mask(values, low, high):
        left = torch.ones_like(values) if low <= 0 else smoothstep((values - low) / feather)
        right = torch.ones_like(values) if high >= 1 else smoothstep((high - values) / feather)
        return left * right * ((values >= low) & (values <= high)).to(values.dtype)

    return (axis_mask(ys, y0, y1)[:, None] * axis_mask(xs, x0, x1)[None, :])[None, None]


def encode_conditions(pipe, max_sequence_length):
    prompts = [AB_PROMPT, A_PROMPT, B_PROMPT]
    positives, positive_pooled = [], []
    negative_embeds = negative_pooled = None
    # Encode each condition independently. In particular, the A+B branch then uses the same
    # text-encoder batch shape as the stock single-prompt pipeline, which is required for the
    # strict manual-loop equivalence control on Hygon fp16 kernels.
    for index, prompt in enumerate(prompts):
        encoded = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        prompt_embed, negative_embed, pooled, negative_pool = encoded
        positives.append(prompt_embed)
        positive_pooled.append(pooled)
        if index == 0:
            negative_embeds = negative_embed
            negative_pooled = negative_pool
    return {
        "positive": torch.cat(positives, dim=0),
        "negative": negative_embeds,
        "positive_pooled": torch.cat(positive_pooled, dim=0),
        "negative_pooled": negative_pooled,
    }


def prepare_schedule(pipe, latents, steps):
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None):
        _, _, height, width = latents.shape
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        steps,
        pipe._execution_device,
        **scheduler_kwargs,
    )
    return timesteps, scheduler_kwargs.get("mu")


def transformer_pair(pipe, latents, timestep, prompt_embeds, pooled_prompt_embeds):
    latent_input = torch.cat([latents, latents], dim=0)
    return pipe.transformer(
        hidden_states=latent_input,
        timestep=timestep.expand(2),
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]


def denoise(pipe, latent, encoded, row, steps, guidance_scale):
    latents = latent.clone()
    timesteps, mu = prepare_schedule(pipe, latents, steps)
    pair_global_embeds = torch.cat([encoded["negative"], encoded["positive"][:1]], dim=0)
    pair_global_pooled = torch.cat(
        [encoded["negative_pooled"], encoded["positive_pooled"][:1]], dim=0
    )
    pair_entity_embeds = encoded["positive"][1:3]
    pair_entity_pooled = encoded["positive_pooled"][1:3]

    plate_mask = soft_box_mask(
        PLATE_BOX, latents.shape[-2], latents.shape[-1], row["feather"], latents.device
    )
    vase_mask = soft_box_mask(
        VASE_BOX, latents.shape[-2], latents.shape[-1], row["feather"], latents.device
    )
    region_weight = float(row["region_weight"])

    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            uncond, global_cond = transformer_pair(
                pipe, latents, timestep, pair_global_embeds, pair_global_pooled
            ).chunk(2)
            global_score = uncond + guidance_scale * (global_cond - uncond)

            if row["active_start"] <= index < min(row["active_end"], steps) and region_weight > 0:
                plate_cond, vase_cond = transformer_pair(
                    pipe, latents, timestep, pair_entity_embeds, pair_entity_pooled
                ).chunk(2)
                plate_score = uncond + guidance_scale * (plate_cond - uncond)
                vase_score = uncond + guidance_scale * (vase_cond - uncond)

                # Weighted least-squares MultiDiffusion fusion. The global path remains everywhere;
                # each region adds only its own prompt's score field on the same current A+B latent.
                numerator = global_score.float()
                numerator = numerator + region_weight * plate_mask * plate_score.float()
                numerator = numerator + region_weight * vase_mask * vase_score.float()
                denominator = 1.0 + region_weight * plate_mask + region_weight * vase_mask
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
        "plate_mask_mean": float(plate_mask.mean().item()),
        "vase_mask_mean": float(vase_mask.mean().item()),
        "interior_own_fraction": region_weight / (1.0 + region_weight),
    }
    return image, diagnostics, plate_mask, vase_mask


def save_mask(mask, path):
    array = (mask[0, 0].detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    Image.fromarray(array, mode="L").resize((1024, 1024), Image.Resampling.NEAREST).save(path)


def compare_images(left, right):
    left = np.asarray(left, dtype=np.int16)
    right = np.asarray(right, dtype=np.int16)
    delta = np.abs(left - right)
    return {
        "mean_abs_pixel_delta": float(delta.mean()),
        "max_abs_pixel_delta": int(delta.max()),
        "different_pixel_channels": int(np.count_nonzero(delta)),
        "total_pixel_channels": int(delta.size),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--equivalence", action="store_true")
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
            row = row
            row["active_end"] = steps
    else:
        steps, height, width, max_sequence_length = 28, 1024, 1024, 256
        run_root = OUT / "formal"

    for sub in ("combined", "equivalence", "analysis", "logs"):
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

    if args.equivalence:
        if not args.smoke:
            raise ValueError("Equivalence control is intentionally restricted to the 512x512 2-step smoke run")
        latent, latent_sha = make_latent(pipe, args.seeds[0], 1, height, width)
        baseline_row = {
            "name": "manual_global_baseline",
            "region_weight": 0.0,
            "active_start": 0,
            "active_end": 0,
            "feather": 0.04,
        }
        manual, diagnostics, _, _ = denoise(
            pipe, latent, encoded, baseline_row, steps, guidance_scale=4.5
        )
        standard = pipe(
            prompt=AB_PROMPT,
            prompt_2=AB_PROMPT,
            prompt_3=AB_PROMPT,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            num_inference_steps=steps,
            guidance_scale=4.5,
            height=height,
            width=width,
            latents=latent.clone(),
            max_sequence_length=max_sequence_length,
            output_type="pil",
        ).images[0]
        stats = compare_images(manual, standard)
        save_png(
            manual,
            run_root / "equivalence" / "manual_global_baseline.png",
            args.seeds[0],
            AB_PROMPT,
            latent_sha,
            {**baseline_row, **diagnostics},
        )
        save_png(
            standard,
            run_root / "equivalence" / "standard_pipeline_baseline.png",
            args.seeds[0],
            AB_PROMPT,
            latent_sha,
            {"name": "standard_pipeline_baseline"},
        )
        (run_root / "equivalence" / "stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        print(json.dumps({"event": "equivalence", **stats}), flush=True)

    for seed in args.seeds:
        latent, latent_sha = make_latent(pipe, seed, 1, height, width)
        for row in rows:
            path = run_root / "combined" / f"seed{seed:06d}__{row['name']}.png"
            if path.exists() and not args.overwrite:
                print(json.dumps({"event": "skipped", "seed": seed, "name": row["name"]}), flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            image, diagnostics, plate_mask, vase_mask = denoise(
                pipe, latent, encoded, row, steps, guidance_scale=4.5
            )
            config = {
                **row,
                **diagnostics,
                "source": "MultiDiffusion/RPG-style regional score fusion adapted to SD3.5",
                "same_current_ab_latent_for_all_scores": True,
                "no_donor_image_or_trajectory": True,
                "boxes": {"plate": PLATE_BOX, "vase": VASE_BOX},
                "guidance_scale": 4.5,
                "steps": steps,
                "height": height,
                "width": width,
            }
            save_png(image, path, seed, AB_PROMPT, latent_sha, config)
            if not (run_root / "analysis" / "plate_mask.png").exists():
                save_mask(plate_mask, run_root / "analysis" / "plate_mask.png")
                save_mask(vase_mask, run_root / "analysis" / "vase_mask.png")
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
