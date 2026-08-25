from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin


SERVER_ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
HELPERS = Path("/science/wx/pry/baseline_v2/lsda_v2_src")
LSDA_CODE = Path("/science/wx/pry/LSDA")
MANIFEST = SERVER_ROOT / "lsda_manifest.json"
OUTPUT = SERVER_ROOT / "lsda_clean"


def native_path(row: dict) -> Path:
    return (
        SERVER_ROOT
        / "base"
        / "images"
        / row["pair_id"]
        / "native_SS"
        / f'{row["native_task_id"]}.png'
    )


def source_mask_dir(row: dict) -> Path:
    # These masks were produced by SAM from the matching native SS image. They are
    # reused as segmentation artifacts only; no old LSDA image/latent is consumed.
    return SERVER_ROOT / "lsda_original" / "masks" / row["task_id"]


def load_exclusive_image_masks(mask_dir: Path) -> tuple[list[np.ndarray], dict]:
    masks = [
        np.asarray(Image.open(mask_dir / f"entity_{index}.png").convert("L")) > 127
        for index in (1, 2)
    ]
    if any(not mask.any() for mask in masks):
        raise RuntimeError(f"empty SAM mask: {mask_dir}")

    raw_overlap = masks[0] & masks[1]
    exclusive = [mask.copy() for mask in masks]
    if raw_overlap.any():
        centers = []
        for mask in masks:
            yy, xx = np.where(mask)
            centers.append((float(xx.mean()), float(yy.mean())))
        yy, xx = np.where(raw_overlap)
        exclusive[0][raw_overlap] = False
        exclusive[1][raw_overlap] = False
        for y, x in zip(yy.tolist(), xx.tolist()):
            winner = int(
                np.argmin([(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in centers])
            )
            exclusive[winner][y, x] = True

    background = ~(exclusive[0] | exclusive[1])
    partition = (
        exclusive[0].astype(np.uint8)
        + exclusive[1].astype(np.uint8)
        + background.astype(np.uint8)
    )
    if int(partition.min()) != 1 or int(partition.max()) != 1:
        raise RuntimeError("image-space ownership is not exactly one-hot")
    summary = {
        "source": "matching native SS image segmented by SAM",
        "mask_dir": str(mask_dir),
        "raw_overlap_fraction": float(raw_overlap.mean()),
        "owner_fractions_image": [float(mask.mean()) for mask in exclusive],
        "background_fraction_image": float(background.mean()),
        "partition_min_image": int(partition.min()),
        "partition_max_image": int(partition.max()),
    }
    return exclusive, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("nshards", type=int)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(LSDA_CODE))
    sys.path.insert(0, str(HELPERS))
    import phase1_common
    phase1_common.MODEL_DIR = MODEL
    from diffusers import StableDiffusion3Pipeline
    from phase212_regional_score import prepare_schedule, transformer_pair
    from phase213_multidiffusion_crop import region_cfg_score
    from phase221_group_overlap_arbitration import encode_prompts
    from lsda_pipeline import (
        GUIDANCE,
        STEPS,
        decode,
        denoise_specialists,
        image_masks_to_latent,
        make_latent,
        native_ss_trajectory,
    )

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    all_tasks = payload["tasks"]
    if len(all_tasks) != 900:
        raise RuntimeError(f"expected 900 tasks, found {len(all_tasks)}")
    tasks = [row for i, row in enumerate(all_tasks) if i % args.nshards == args.shard]
    if args.limit:
        tasks = tasks[: args.limit]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_log = log_dir / f"failures_shard_{args.shard}.jsonl"

    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (prepare_schedule, transformer_pair, region_cfg_score)

    for row in tasks:
        stem = row["task_id"].replace("lsda_ra", "lsda_clean")
        image_dir = OUTPUT / "images" / row["pair_id"]
        sidecar_dir = OUTPUT / "sidecars" / row["pair_id"]
        image_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{stem}.png"
        sidecar_path = sidecar_dir / f"{stem}.json"
        if image_path.exists() and sidecar_path.exists():
            print(json.dumps({"event": "skip", "task_id": stem}), flush=True)
            continue

        started = time.time()
        try:
            mask_dir = source_mask_dir(row)
            owner_images, mask_audit = load_exclusive_image_masks(mask_dir)
            native_image = native_path(row)
            if not native_image.exists():
                raise FileNotFoundError(native_image)

            prompts = (
                row["global_prompt"],
                row["entity_A_prompt"],
                row["entity_B_prompt"],
            )
            encoded = encode_prompts(pipe, prompts)
            initial, latent_sha = make_latent(pipe, int(row["latent_seed"]))
            native_states, _, native_mu = native_ss_trajectory(
                pipe, initial, encoded, helpers
            )
            owners, background = image_masks_to_latent(owner_images, initial)
            final_latent, denoise_audit = denoise_specialists(
                pipe,
                initial,
                encoded,
                owners,
                background,
                native_states,
                helpers,
            )
            image = decode(pipe, final_latent)

            record = {
                **row,
                "task_id": stem,
                "method": "LSDA clean strict one-hot local specialists from step 0",
                "output_version": "lsda_clean_v1",
                "steps": STEPS,
                "guidance_scale": GUIDANCE,
                "entity_prompts": [row["entity_A_prompt"], row["entity_B_prompt"]],
                "entity_prompt_type": "standalone Short phrase",
                "region_source": row["native_task_id"],
                "region_source_type": "native SS SAM contour",
                "background_expert": "same-seed native SS trajectory, complement only",
                "standalone_image_latent_or_hidden_used": False,
                "sl_ll_image_mask_latent_or_hidden_used": False,
                "old_lsda_result_used": False,
                "latent_sha256": latent_sha,
                "native_ss_png_sha256": sha256_file(native_image),
                "native_mu": native_mu,
                "segmentation": mask_audit,
                "denoising": denoise_audit,
                "seconds": time.time() - started,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            info = PngImagePlugin.PngInfo()
            info.add_text("task_id", stem)
            info.add_text("latent_seed", str(row["latent_seed"]))
            info.add_text("method", record["method"])
            image.save(image_path, pnginfo=info)
            sidecar_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"event": "done", "task_id": stem, "seconds": record["seconds"]}
                ),
                flush=True,
            )
            del encoded, initial, native_states, owners, background, final_latent, image
        except Exception as exc:
            failure = {
                "task_id": stem,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "failed", **failure}, ensure_ascii=False), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
