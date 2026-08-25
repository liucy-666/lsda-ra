from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import PngImagePlugin


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MANIFEST = ROOT / "lsda_manifest.json"
MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
SRC = Path("/science/wx/pry/baseline_v2/lsda_v2_src")
STEPS = 28
HANDOFF = 7
GUIDANCE = 4.5
EARLY_WEIGHTS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00)
BACKGROUND = "neutral studio background and surface only, without objects"


def make_latent(pipe, seed: int):
    generator = torch.Generator("cuda").manual_seed(seed)
    latent = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, 1024, 1024,
        torch.float16, torch.device("cuda"), generator, None,
    )
    digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    return latent, digest


def local_score(pipe, latents, timestep, encoded, prompt_index, bounds, region_cfg_score):
    left, top, right, bottom = bounds
    crop = latents[:, :, top:bottom, left:right]
    return region_cfg_score(
        pipe, crop, timestep,
        encoded["negative"], encoded["positive"][prompt_index:prompt_index + 1],
        encoded["negative_pooled"], encoded["positive_pooled"][prompt_index:prompt_index + 1],
        GUIDANCE,
    ).float()


def denoise(pipe, initial, encoded, mask_dir, helpers):
    prepare_schedule, cfg_score, region_cfg_score, load_mask, make_exclusive_owners, owner_bounds = helpers
    latents = initial.clone()
    timesteps, mu = prepare_schedule(pipe, latents, STEPS)
    height, width = latents.shape[-2:]
    patch_size = int(pipe.transformer.config.patch_size)
    masks = [
        load_mask(mask_dir / "entity_1.png", height, width, latents.device),
        load_mask(mask_dir / "entity_2.png", height, width, latents.device),
    ]
    owners, conflict, background_owner = make_exclusive_owners(masks)
    bounds = [owner_bounds(owner, patch_size) for owner in owners]
    soft = [
        F.avg_pool2d(F.max_pool2d(owner, 5, 1, 2), 7, 1, 3).clamp(0, 1)
        for owner in owners
    ]

    with torch.inference_mode():
        for step, timestep in enumerate(timesteps):
            if step < HANDOFF:
                global_velocity = cfg_score(pipe, latents, timestep, encoded, 0).float()
                velocity = global_velocity.clone()
                weight = EARLY_WEIGHTS[step]
                for prompt_index, (soft_owner, region) in enumerate(zip(soft, bounds), start=1):
                    left, top, right, bottom = region
                    local = local_score(
                        pipe, latents, timestep, encoded, prompt_index, region, region_cfg_score
                    )
                    velocity[:, :, top:bottom, left:right] += (
                        weight * soft_owner[:, :, top:bottom, left:right]
                        * (local - global_velocity[:, :, top:bottom, left:right])
                    )
            else:
                velocity = background_owner * cfg_score(pipe, latents, timestep, encoded, 3).float()
                for prompt_index, (owner, region) in enumerate(zip(owners, bounds), start=1):
                    left, top, right, bottom = region
                    local = local_score(
                        pipe, latents, timestep, encoded, prompt_index, region, region_cfg_score
                    )
                    velocity[:, :, top:bottom, left:right] += (
                        owner[:, :, top:bottom, left:right] * local
                    )
            latents = pipe.scheduler.step(
                velocity.to(latents.dtype), timestep, latents, return_dict=False
            )[0].to(initial.dtype)

        scaled = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return image, {
        "mu": mu,
        "owner_fractions": [float(x.mean().item()) for x in owners],
        "background_fraction": float(background_owner.mean().item()),
        "conflict_fraction": float(conflict.mean().item()),
        "bounds": bounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("nshards", type=int)
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
    tasks = [row for i, row in enumerate(payload["tasks"]) if i % args.nshards == args.shard]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (prepare_schedule, cfg_score, region_cfg_score, load_mask,
               make_exclusive_owners, owner_bounds)

    for row in tasks:
        image_dir = ROOT / "lsda" / "images" / row["pair_id"]
        sidecar_dir = ROOT / "lsda" / "sidecars" / row["pair_id"]
        image_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f'{row["task_id"]}.png'
        sidecar_path = sidecar_dir / f'{row["task_id"]}.json'
        if image_path.exists() and sidecar_path.exists():
            print(json.dumps({"event": "skip", "task_id": row["task_id"]}), flush=True)
            continue

        prompts = (row["global_prompt"], row["entity_A_prompt"], row["entity_B_prompt"], BACKGROUND)
        encoded = encode_prompts(pipe, prompts)
        initial, latent_sha = make_latent(pipe, int(row["latent_seed"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        image, diagnostics = denoise(
            pipe, initial, encoded, ROOT / "lsda" / "masks" / row["task_id"], helpers
        )
        record = {
            **row,
            "method": "LSDA-RA standalone-short early-entry one-hot handoff",
            "external_visual_sources": [row["standalone_A_task_id"], row["standalone_B_task_id"]],
            "external_visual_information": "standalone Short masks only",
            "sl_ll_information_used": False,
            "background_prompt": BACKGROUND,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "handoff_step": HANDOFF,
            "early_local_weights": list(EARLY_WEIGHTS),
            "latent_sha256": latent_sha,
            "seconds": time.time() - started,
            "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "diagnostics": diagnostics,
        }
        info = PngImagePlugin.PngInfo()
        info.add_text("task_id", row["task_id"])
        info.add_text("prompt", row["global_prompt"])
        info.add_text("latent_seed", str(row["latent_seed"]))
        info.add_text("metadata", json.dumps(record, ensure_ascii=False, sort_keys=True))
        image.save(image_path, pnginfo=info)
        sidecar_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", "task_id": row["task_id"],
                          "seconds": record["seconds"]}), flush=True)
        del encoded, initial, image
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
