from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import PngImagePlugin


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MANIFEST = ROOT / "lsda_manifest.json"
MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
SRC = Path("/science/wx/pry/baseline_v2/lsda_v2_src")
OUT = ROOT / "lsda_independent_pilot"
STEPS = 28
GUIDANCE = 4.5


def make_latent(pipe, seed: int):
    generator = torch.Generator("cuda").manual_seed(seed)
    latent = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        1024,
        1024,
        torch.float16,
        torch.device("cuda"),
        generator,
        None,
    )
    digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    return latent, digest


def local_velocity(pipe, latent, timestep, encoded, prompt_index, region_cfg_score):
    return region_cfg_score(
        pipe,
        latent,
        timestep,
        encoded["negative"],
        encoded["positive"][prompt_index : prompt_index + 1],
        encoded["negative_pooled"],
        encoded["positive_pooled"][prompt_index : prompt_index + 1],
        GUIDANCE,
    )


def decode(pipe, latents):
    scaled = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(scaled, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def denoise(pipe, initial, encoded, mask_dir, helpers):
    prepare_schedule, cfg_score, region_cfg_score, load_mask, make_exclusive_owners, owner_bounds = helpers
    global_latent = initial.clone()
    timesteps, mu = prepare_schedule(pipe, global_latent, STEPS)
    global_scheduler = copy.deepcopy(pipe.scheduler)

    full_h, full_w = global_latent.shape[-2:]
    patch_size = int(pipe.transformer.config.patch_size)
    masks = [
        load_mask(mask_dir / "entity_1.png", full_h, full_w, global_latent.device),
        load_mask(mask_dir / "entity_2.png", full_h, full_w, global_latent.device),
    ]
    owners, conflict, background = make_exclusive_owners(masks)
    bounds = [owner_bounds(owner, patch_size, margin=6) for owner in owners]

    local_latents = []
    local_masks = []
    local_schedulers = []
    for owner, (left, top, right, bottom) in zip(owners, bounds):
        local_latents.append(initial[:, :, top:bottom, left:right].clone())
        local_masks.append(owner[:, :, top:bottom, left:right])
        local_schedulers.append(copy.deepcopy(pipe.scheduler))

    with torch.inference_mode():
        for timestep in timesteps:
            global_velocity = cfg_score(pipe, global_latent, timestep, encoded, 0)
            global_next = global_scheduler.step(
                global_velocity.to(global_latent.dtype), timestep, global_latent, return_dict=False
            )[0].to(initial.dtype)

            next_locals = []
            for prompt_index, (local, local_mask, scheduler, region) in enumerate(
                zip(local_latents, local_masks, local_schedulers, bounds), start=1
            ):
                velocity = local_velocity(
                    pipe, local, timestep, encoded, prompt_index, region_cfg_score
                )
                local_next = scheduler.step(
                    velocity.to(local.dtype), timestep, local, return_dict=False
                )[0].to(initial.dtype)
                left, top, right, bottom = region
                global_context = global_next[:, :, top:bottom, left:right]
                # The entity interior keeps its private trajectory. Only pixels outside
                # the immutable SS contour are synchronized to the SS scaffold.
                local_next = (
                    local_mask * local_next + (1.0 - local_mask) * global_context
                ).to(initial.dtype)
                next_locals.append(local_next)

            global_latent = global_next
            local_latents = next_locals

        composed = background * global_latent
        for owner, local, (left, top, right, bottom) in zip(owners, local_latents, bounds):
            owner_crop = owner[:, :, top:bottom, left:right]
            composed[:, :, top:bottom, left:right] += owner_crop * local

        composed = composed.to(initial.dtype)
        image = decode(pipe, composed)
        scaffold = decode(pipe, global_latent)

    diagnostics = {
        "mu": mu,
        "steps": STEPS,
        "private_entity_latents": True,
        "private_trajectory_start_step": 0,
        "ss_latent_inside_entity_used": False,
        "ss_information_used": "SAM contour, background, position, scale, and occlusion only",
        "entity_prompt_type": "standalone Short phrase",
        "owner_bounds_latent": bounds,
        "owner_fractions": [float(owner.mean().item()) for owner in owners],
        "background_fraction": float(background.mean().item()),
        "conflict_fraction": float(conflict.mean().item()),
    }
    return image, scaffold, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-ids", nargs="+", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(SRC))
    import phase1_common
    phase1_common.MODEL_DIR = MODEL
    from diffusers import StableDiffusion3Pipeline
    from phase212_regional_score import prepare_schedule
    from phase213_multidiffusion_crop import region_cfg_score
    from phase221_group_overlap_arbitration import encode_prompts
    from phase224_structure_locked_appearance import load_mask, make_exclusive_owners, owner_bounds
    from phase225_irreversible_handoff import cfg_score

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    index = {row["task_id"]: row for row in payload["tasks"]}
    rows = [index[task_id] for task_id in args.task_ids]

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (
        prepare_schedule,
        cfg_score,
        region_cfg_score,
        load_mask,
        make_exclusive_owners,
        owner_bounds,
    )

    for row in rows:
        output_id = row["task_id"].replace("lsda_ra", "lsda_independent")
        # The original-SS SAM masks retain the lsda_ra manifest task id as their directory name.
        mask_dir = ROOT / "lsda_original" / "masks" / row["task_id"]
        image_dir = OUT / "images" / row["pair_id"]
        scaffold_dir = OUT / "scaffolds" / row["pair_id"]
        sidecar_dir = OUT / "sidecars" / row["pair_id"]
        for directory in (image_dir, scaffold_dir, sidecar_dir):
            directory.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{output_id}.png"
        scaffold_path = scaffold_dir / f"{output_id}__ss_scaffold.png"
        sidecar_path = sidecar_dir / f"{output_id}.json"

        prompts = (
            row["global_prompt"],
            row["entity_A_prompt"],
            row["entity_B_prompt"],
        )
        encoded = encode_prompts(pipe, prompts)
        initial, latent_sha = make_latent(pipe, int(row["latent_seed"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        image, scaffold, diagnostics = denoise(pipe, initial, encoded, mask_dir, helpers)
        record = {
            **row,
            "task_id": output_id,
            "method": "LSDA independent entity latent pilot",
            "mask_source": str(mask_dir),
            "standalone_visual_information_used": False,
            "sl_ll_information_used": False,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "latent_sha256": latent_sha,
            "seconds": time.time() - started,
            "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "diagnostics": diagnostics,
        }
        info = PngImagePlugin.PngInfo()
        info.add_text("task_id", output_id)
        info.add_text("prompt", row["global_prompt"])
        info.add_text("metadata", json.dumps(record, ensure_ascii=False, sort_keys=True))
        image.save(image_path, pnginfo=info)
        scaffold.save(scaffold_path)
        sidecar_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", "task_id": output_id, "seconds": record["seconds"]}), flush=True)
        del encoded, initial, image, scaffold
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
