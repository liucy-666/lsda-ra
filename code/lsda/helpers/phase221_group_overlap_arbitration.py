from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline

from phase28_trainfree import MODEL_DIR, ROOT, make_latent, save_png
from phase212_regional_score import prepare_schedule, soft_box_mask, transformer_pair
from phase213_multidiffusion_crop import crop_bounds, region_cfg_score


OUT = ROOT / "phase221_group_overlap_arbitration"
REGION_WEIGHT = 3.0
JOINT_WEIGHT = 3.0
ACTIVE_START = 8
ACTIVE_END = 22
FEATHER = 0.05


@dataclass(frozen=True)
class Scenario:
    name: str
    global_prompt: str
    entity_prompts: tuple[str, ...]
    crops: tuple[tuple[float, float, float, float], ...]
    joint_crop: tuple[float, float, float, float]


SCENARIOS = {
    "hug": Scenario(
        name="hug",
        global_prompt=(
            "A joyful Black man and a white girl are warmly hugging each other, "
            "natural affectionate body language, casual clothing, photorealistic."
        ),
        entity_prompts=(
            "A single joyful Black man in casual clothing, seen from the waist up, "
            "arms extended in a natural hugging pose, photorealistic.",
            "A single joyful white girl in casual clothing, seen from the waist up, "
            "arms extended in a natural hugging pose, photorealistic.",
        ),
        crops=((0.08, 0.05, 0.70, 0.98), (0.30, 0.05, 0.92, 0.98)),
        joint_crop=(0.08, 0.05, 0.92, 0.98),
    ),
    "cats_sofa": Scenario(
        name="cats_sofa",
        global_prompt=(
            "A gray short-haired cat and a Maine Coon cat are curled up together "
            "on an elegant Italian-style sofa in a cozy room, photorealistic."
        ),
        entity_prompts=(
            "A single gray short-haired cat curled up comfortably, photorealistic.",
            "A single fluffy Maine Coon cat curled up comfortably, photorealistic.",
            "A single elegant Italian-style upholstered sofa with carved wooden details "
            "in a cozy room, photorealistic.",
        ),
        crops=(
            (0.08, 0.20, 0.62, 0.82),
            (0.38, 0.18, 0.92, 0.82),
            (0.02, 0.28, 0.98, 0.98),
        ),
        joint_crop=(0.02, 0.15, 0.98, 0.98),
    ),
}


def encode_prompts(pipe, prompts, max_sequence_length=256):
    positives, positive_pooled = [], []
    negative_embeds = negative_pooled = None
    for index, prompt in enumerate(prompts):
        prompt_embed, negative_embed, pooled, negative_pool = pipe.encode_prompt(
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


def pairwise_ambiguity(masks):
    """Soft union of all pairwise mask intersections; scales to any entity count."""
    pair_supports = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            pair_supports.append(torch.minimum(masks[i], masks[j]))
    if not pair_supports:
        return torch.zeros_like(masks[0])
    remaining = torch.ones_like(pair_supports[0])
    for support in pair_supports:
        remaining = remaining * (1.0 - support)
    return 1.0 - remaining


def save_mask(mask, path):
    array = np.round(mask[0, 0].detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(array, mode="L").resize((1024, 1024), Image.Resampling.NEAREST).save(path)


def denoise(pipe, latent, encoded, scenario, intervention, steps=28, guidance_scale=4.5):
    latents = latent.clone()
    timesteps, mu = prepare_schedule(pipe, latents, steps)
    patch_size = int(pipe.transformer.config.patch_size)
    full_h, full_w = latents.shape[-2:]
    bounds = [crop_bounds(box, full_h, full_w, patch_size) for box in scenario.crops]
    masks = [soft_box_mask(box, full_h, full_w, FEATHER, latents.device) for box in scenario.crops]
    ambiguity = pairwise_ambiguity(masks)
    joint_bounds = crop_bounds(scenario.joint_crop, full_h, full_w, patch_size)
    jl, jt, jr, jb = joint_bounds
    joint_support = torch.zeros_like(ambiguity)
    joint_support[:, :, jt:jb, jl:jr] = 1.0
    ambiguity = ambiguity * joint_support

    global_embeds = torch.cat([encoded["negative"], encoded["positive"][:1]], dim=0)
    global_pooled = torch.cat(
        [encoded["negative_pooled"], encoded["positive_pooled"][:1]], dim=0
    )

    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            uncond, global_cond = transformer_pair(
                pipe, latents, timestep, global_embeds, global_pooled
            ).chunk(2)
            global_score = uncond + guidance_scale * (global_cond - uncond)
            active = intervention and ACTIVE_START <= index < ACTIVE_END
            local_scale = (1.0 - ambiguity) if active else torch.ones_like(ambiguity)
            numerator = global_score.float()
            denominator = torch.ones(
                (1, 1, full_h, full_w), device=latents.device, dtype=torch.float32
            )

            for entity_index, (entity_bounds, mask) in enumerate(zip(bounds, masks), start=1):
                left, top, right, bottom = entity_bounds
                crop = latents[:, :, top:bottom, left:right]
                local_score = region_cfg_score(
                    pipe,
                    crop,
                    timestep,
                    encoded["negative"],
                    encoded["positive"][entity_index:entity_index + 1],
                    encoded["negative_pooled"],
                    encoded["positive_pooled"][entity_index:entity_index + 1],
                    guidance_scale,
                )
                effective_mask = (
                    mask[:, :, top:bottom, left:right]
                    * local_scale[:, :, top:bottom, left:right]
                )
                numerator[:, :, top:bottom, left:right] += (
                    REGION_WEIGHT * effective_mask * local_score.float()
                )
                denominator[:, :, top:bottom, left:right] += REGION_WEIGHT * effective_mask

            if active:
                joint_latent = latents[:, :, jt:jb, jl:jr]
                joint_score = region_cfg_score(
                    pipe,
                    joint_latent,
                    timestep,
                    encoded["negative"],
                    encoded["positive"][:1],
                    encoded["negative_pooled"],
                    encoded["positive_pooled"][:1],
                    guidance_scale,
                )
                support = ambiguity[:, :, jt:jb, jl:jr]
                numerator[:, :, jt:jb, jl:jr] += JOINT_WEIGHT * support * joint_score.float()
                denominator[:, :, jt:jb, jl:jr] += JOINT_WEIGHT * support

            velocity = (numerator / denominator).to(global_score.dtype)
            previous_dtype = latents.dtype
            latents = pipe.scheduler.step(velocity, timestep, latents, return_dict=False)[0]
            if latents.dtype != previous_dtype:
                latents = latents.to(previous_dtype)

        scaled = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    diagnostics = {
        "mu": mu,
        "timesteps": len(timesteps),
        "entity_count": len(scenario.entity_prompts),
        "entity_crops": scenario.crops,
        "entity_bounds_latent": bounds,
        "joint_crop": scenario.joint_crop,
        "joint_bounds_latent": joint_bounds,
        "mean_ambiguity_support": float(ambiguity.mean().item()),
        "active_start": ACTIVE_START if intervention else None,
        "active_end_exclusive": ACTIVE_END if intervention else None,
        "region_weight": REGION_WEIGHT,
        "joint_weight": JOINT_WEIGHT if intervention else 0.0,
        "arbitration": (
            "suppress independent entity scores in pairwise-overlap support and add group joint score"
            if intervention else "unmodified local-canvas least-squares fusion"
        ),
        "additional_model_forwards": (ACTIVE_END - ACTIVE_START) if intervention else 0,
    }
    return image, diagnostics, ambiguity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]
    run_root = OUT / "formal" / scenario.name
    image_dir = run_root / "images"
    analysis_dir = run_root / "analysis"
    for directory in (image_dir, analysis_dir, run_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, local_files_only=True
    )
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    encoded = encode_prompts(pipe, (scenario.global_prompt,) + scenario.entity_prompts)
    latent, latent_sha = make_latent(pipe, args.seed, 1, 1024, 1024)

    records = []
    for intervention in (False, True):
        mode = "intervened" if intervention else "original"
        output_path = image_dir / f"seed{args.seed:06d}__{scenario.name}__{mode}.png"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(output_path)
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        image, diagnostics, ambiguity = denoise(pipe, latent, encoded, scenario, intervention)
        seconds = time.time() - started
        config = {
            **diagnostics,
            "source": "Phase 2.21 generalized group-overlap arbitration",
            "scenario": scenario.name,
            "global_prompt": scenario.global_prompt,
            "entity_prompts": scenario.entity_prompts,
            "seed": args.seed,
            "steps": 28,
            "height": 1024,
            "width": 1024,
            "guidance_scale": 4.5,
            "same_initial_latent_for_pair": True,
            "no_donor_image_or_trajectory": True,
        }
        save_png(image, output_path, args.seed, scenario.global_prompt, latent_sha, config)
        record = {
            "event": "completed",
            "scenario": scenario.name,
            "mode": mode,
            "seconds": seconds,
            "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "diagnostics": diagnostics,
        }
        records.append(record)
        print(json.dumps(record), flush=True)

    save_mask(ambiguity, analysis_dir / "pairwise_ambiguity_support.png")
    (analysis_dir / "run_summary.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
