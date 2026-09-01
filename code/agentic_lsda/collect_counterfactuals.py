from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin

from schema import validate_manifest
from timed_lsda import denoise_timed_specialists


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_png(image: Image.Image, path: Path, metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = PngImagePlugin.PngInfo()
    for key, value in metadata.items():
        info.add_text(key, value)
    image.save(path, pnginfo=info)


def native_image_path(root: Path, task: dict) -> Path:
    candidates = [
        root / task["pair_id"] / "native_SS" / f'{task["native_task_id"]}.png',
        *[root / f'{task["sample_id"]}{suffix}' for suffix in (".png", ".jpg", ".jpeg", ".webp")],
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_npz_masks(root: Path, task: dict) -> tuple[list[np.ndarray], dict]:
    path = root / f'{task["sample_id"]}_mask.npz'
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as payload:
        if not {"left", "right"}.issubset(payload.files):
            raise ValueError(f"mask archive lacks left/right arrays: {path}")
        masks = [payload["left"].astype(bool), payload["right"].astype(bool)]
    if masks[0].shape != masks[1].shape or any(not mask.any() for mask in masks):
        raise ValueError(f"invalid or empty masks: {path}")
    raw_overlap = masks[0] & masks[1]
    if raw_overlap.any():
        raise ValueError(f"packaged masks are not exclusive: {path}")
    background = ~(masks[0] | masks[1])
    partition = masks[0].astype(np.uint8) + masks[1].astype(np.uint8) + background.astype(np.uint8)
    if int(partition.min()) != 1 or int(partition.max()) != 1:
        raise ValueError(f"packaged masks are not one-hot: {path}")
    return masks, {
        "source": "matching native SS image segmented by SAM; packaged immutable NPZ",
        "mask_archive": str(path),
        "mask_archive_sha256": sha256_file(path),
        "owner_fractions_image": [float(mask.mean()) for mask in masks],
        "background_fraction_image": float(background.mean()),
        "partition_min_image": int(partition.min()),
        "partition_max_image": int(partition.max()),
    }


def checkpoint_before_step(initial: torch.Tensor, native_states: list[torch.Tensor], step: int) -> torch.Tensor:
    return initial if step == 0 else native_states[step - 1]


def group_tasks(tasks: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        groups[task["sample_id"]].append(task)
    return [groups[key] for key in sorted(groups)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--native-images-root", type=Path, required=True)
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument("--mask-root", type=Path)
    mask_group.add_argument("--mask-npz-root", type=Path)
    parser.add_argument("--image-output-dir", type=Path, required=True)
    parser.add_argument("--record-output-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, path in (
        ("manifest", args.manifest),
        ("model", args.model_dir),
        ("native images", args.native_images_root),
        ("SAM masks", args.mask_root or args.mask_npz_root),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if not 0 <= args.shard < args.nshards:
        raise ValueError("shard must satisfy 0 <= shard < nshards")

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(payload)
    generation = payload["generation"]
    if int(generation["num_inference_steps"]) != 28:
        raise ValueError("the current LSDA v1 runtime is frozen to 28 steps")

    sample_groups = group_tasks(payload["tasks"])
    sample_groups = [group for index, group in enumerate(sample_groups) if index % args.nshards == args.shard]
    if args.limit_samples:
        sample_groups = sample_groups[: args.limit_samples]

    args.image_output_dir.mkdir(parents=True, exist_ok=True)
    args.record_output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_root = args.record_output_dir / "sidecars"
    checkpoint_record_root = args.record_output_dir / "checkpoints"
    log_root = args.record_output_dir / "logs"
    sidecar_root.mkdir(parents=True, exist_ok=True)
    checkpoint_record_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    failure_log = log_root / f"failures_shard_{args.shard}.jsonl"

    lsda_dir = Path(__file__).resolve().parents[1] / "lsda"
    sys.path.insert(0, str(lsda_dir))
    from diffusers import StableDiffusion3Pipeline
    from generate_lsda_900 import load_exclusive_image_masks
    from lsda_pipeline import (
        decode,
        image_masks_to_latent,
        make_latent,
        native_ss_trajectory,
        owner_bounds,
    )
    from runtime import encode_prompts, prepare_schedule, region_cfg_score, transformer_pair

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    native_helpers = (prepare_schedule, transformer_pair, region_cfg_score)
    timed_helpers = (prepare_schedule, transformer_pair, region_cfg_score, owner_bounds)

    for group in sample_groups:
        base = group[0]
        sample_id = base["sample_id"]
        started = time.time()
        try:
            if any(task["sample_id"] != sample_id for task in group):
                raise RuntimeError("mixed sample group")
            prompts = (base["global_prompt"], base["entity_A_prompt"], base["entity_B_prompt"])
            encoded = encode_prompts(pipe, prompts)
            initial, latent_sha = make_latent(pipe, int(base["latent_seed"]))
            native_states, _, native_mu = native_ss_trajectory(pipe, initial, encoded, native_helpers)

            if args.mask_npz_root:
                owner_images, mask_audit = load_npz_masks(args.mask_npz_root, base)
            else:
                source_mask_dir = args.mask_root / base["source_lsda_task_id"]
                owner_images, mask_audit = load_exclusive_image_masks(source_mask_dir)
            owners, background = image_masks_to_latent(owner_images, initial)
            source_native = native_image_path(args.native_images_root, base)
            if not source_native.exists():
                raise FileNotFoundError(source_native)

            intervention_steps = sorted(
                {
                    int(task["action"]["intervention_step"])
                    for task in group
                    if task["action"]["kind"] == "invoke_lsda"
                }
            )
            checkpoint_index: dict[str, dict] = {}
            for step in intervention_steps:
                checkpoint = checkpoint_before_step(initial, native_states, step)
                checkpoint_image_path = (
                    args.image_output_dir / sample_id / "checkpoints" / f"native_before_t{step:02d}.png"
                )
                checkpoint_record_path = checkpoint_record_root / sample_id / f"native_before_t{step:02d}.json"
                checkpoint_hash = sha256_tensor(checkpoint)
                if args.overwrite or not checkpoint_image_path.exists():
                    save_png(
                        decode(pipe, checkpoint),
                        checkpoint_image_path,
                        {
                            "sample_id": sample_id,
                            "decision_step": str(step),
                            "state": "native_before_intervention",
                            "latent_sha256": checkpoint_hash,
                        },
                    )
                checkpoint_record = {
                    "sample_id": sample_id,
                    "decision_step": step,
                    "latent_sha256": checkpoint_hash,
                    "image_path": str(checkpoint_image_path),
                    "online_internal_metrics": None,
                    "missing_metric_reason": "MM-DiT MLP/attention/residual hooks are not yet implemented",
                }
                checkpoint_record_path.parent.mkdir(parents=True, exist_ok=True)
                if args.overwrite or not checkpoint_record_path.exists():
                    checkpoint_record_path.write_text(
                        json.dumps(checkpoint_record, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                checkpoint_index[str(step)] = checkpoint_record

            for task in group:
                action = task["action"]
                replay_id = task["replay_id"]
                sidecar_path = sidecar_root / sample_id / f"{replay_id}.json"
                if action["kind"] == "wait":
                    record = {
                        **task,
                        "status": "reference_only",
                        "source_native_image": str(source_native),
                        "source_native_sha256": sha256_file(source_native),
                        "latent_sha256": latent_sha,
                        "native_mu": native_mu,
                        "segmentation": mask_audit,
                        "online_internal_metrics": None,
                    }
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                    if args.overwrite or not sidecar_path.exists():
                        sidecar_path.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                    continue

                image_path = args.image_output_dir / sample_id / "candidates" / f"{replay_id}.png"
                if image_path.exists() and sidecar_path.exists() and not args.overwrite:
                    print(json.dumps({"event": "skip", "replay_id": replay_id}), flush=True)
                    continue
                torch.cuda.reset_peak_memory_stats()
                branch_started = time.time()
                final_latent, branch_audit = denoise_timed_specialists(
                    pipe=pipe,
                    initial=initial,
                    encoded=encoded,
                    owners=owners,
                    background=background,
                    native_states=native_states,
                    helper_functions=timed_helpers,
                    intervention_step=int(action["intervention_step"]),
                    target=action["target"],
                    guidance_scale=float(generation["guidance_scale"]),
                )
                save_png(
                    decode(pipe, final_latent),
                    image_path,
                    {
                        "replay_id": replay_id,
                        "sample_id": sample_id,
                        "intervention_step": str(action["intervention_step"]),
                        "target": action["target"],
                    },
                )
                record = {
                    **task,
                    "status": "generated",
                    "image_path": str(image_path),
                    "source_native_image": str(source_native),
                    "source_native_sha256": sha256_file(source_native),
                    "latent_sha256": latent_sha,
                    "native_mu": native_mu,
                    "checkpoint": checkpoint_index[str(action["intervention_step"])],
                    "segmentation": mask_audit,
                    "denoising": branch_audit,
                    "seconds": time.time() - branch_started,
                    "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "binding_score": None,
                    "structure_score": None,
                    "oracle_eligible": False,
                    "oracle_blocker": "binding and structure scores have not been computed",
                }
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                sidecar_path.write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "event": "generated",
                            "replay_id": replay_id,
                            "seconds": record["seconds"],
                        }
                    ),
                    flush=True,
                )
                del final_latent

            print(
                json.dumps(
                    {"event": "sample_complete", "sample_id": sample_id, "seconds": time.time() - started}
                ),
                flush=True,
            )
            del encoded, initial, native_states, owners, background
        except Exception as exc:
            failure = {
                "sample_id": sample_id,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            with failure_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "failed", **failure}, ensure_ascii=False), flush=True)
        finally:
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
