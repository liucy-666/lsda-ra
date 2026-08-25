from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import SamModel, SamProcessor


MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
SAM_MODEL = Path("/science/wx/pry/baseline_v2/models/sam-vit-base")
HELPERS = Path("/science/wx/pry/baseline_v2/lsda_v2_src")
ROOT = Path("/science/wx/pry/LSDA")
STEPS = 28
GUIDANCE = 4.5
SIZE = 1024


@dataclass
class Candidate:
    mask: np.ndarray
    point: tuple[int, int]
    sam_iou: float
    score: float
    area: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


def make_latent(pipe, seed: int) -> tuple[torch.Tensor, str]:
    generator = torch.Generator("cuda").manual_seed(seed)
    latent = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        SIZE,
        SIZE,
        torch.float16,
        torch.device("cuda"),
        generator,
        None,
    )
    digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    return latent, digest


def decode(pipe, latent: torch.Tensor) -> Image.Image:
    with torch.inference_mode():
        scaled = latent.detach() / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
        decoded = pipe.vae.decode(scaled, return_dict=False)[0].detach()
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def expected_x(position: str, index: int, count: int) -> float:
    named = {"left": 0.27, "center": 0.50, "right": 0.73}
    if position in named:
        return named[position]
    return (index + 1) / (count + 1)


def salience_points(image: Image.Image, x_center: float, count: int = 8) -> list[tuple[int, int]]:
    small = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32)
    border = np.concatenate(
        [small[:5].reshape(-1, 3), small[-5:].reshape(-1, 3),
         small[:, :5].reshape(-1, 3), small[:, -5:].reshape(-1, 3)], axis=0
    )
    bg = np.median(border, axis=0)
    color = np.linalg.norm(small - bg[None, None], axis=2) / 255.0
    gx = np.linalg.norm(np.diff(small, axis=1, prepend=small[:, :1]), axis=2) / 255.0
    gy = np.linalg.norm(np.diff(small, axis=0, prepend=small[:1]), axis=2) / 255.0
    score = color + 0.45 * (gx + gy)
    xs = np.arange(128)[None, :] / 127.0
    ys = np.arange(128)[:, None] / 127.0
    score -= 1.2 * np.abs(xs - x_center)
    score -= 0.25 * np.abs(ys - 0.55)
    score[(np.arange(128) / 127.0 < 0.10) | (np.arange(128) / 127.0 > 0.92), :] = -1e9
    score[:, np.abs(np.arange(128) / 127.0 - x_center) > 0.28] = -1e9

    points: list[tuple[int, int]] = []
    work = score.copy()
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(work), work.shape)
        if not np.isfinite(work[y, x]):
            break
        points.append((int((x + 0.5) * 8), int((y + 0.5) * 8)))
        yy, xx = np.ogrid[:128, :128]
        work[(xx - x) ** 2 + (yy - y) ** 2 < 10 ** 2] = -1e9

    anchors = [
        (round(x_center * SIZE), round(0.50 * SIZE)),
        (round(x_center * SIZE), round(0.63 * SIZE)),
        (round(x_center * SIZE), round(0.37 * SIZE)),
    ]
    for point in anchors:
        if point not in points:
            points.append(point)
    return points


def nearest_true(mask: np.ndarray, point: tuple[int, int], radius: int = 48):
    x, y = point
    h, w = mask.shape
    if 0 <= x < w and 0 <= y < h and mask[y, x]:
        return x, y
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    yy, xx = np.where(mask[y0:y1, x0:x1])
    if not len(xx):
        return None
    xx, yy = xx + x0, yy + y0
    j = int(np.argmin((xx - x) ** 2 + (yy - y) ** 2))
    return int(xx[j]), int(yy[j])


def seeded_component(mask: np.ndarray, point: tuple[int, int]) -> np.ndarray:
    seed = nearest_true(mask, point)
    if seed is None:
        return np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    sx, sy = seed
    out = np.zeros_like(mask, dtype=bool)
    out[sy, sx] = True
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                queue.append((nx, ny))
    return out


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    queue = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not exterior[y, x]:
                exterior[y, x] = True
                queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx] and not exterior[ny, nx]:
                exterior[ny, nx] = True
                queue.append((nx, ny))
    return mask | (~mask & ~exterior)


def mask_geometry(mask: np.ndarray):
    ys, xs = np.where(mask)
    if not len(xs):
        return 0.0, (0, 0, 0, 0), (0.0, 0.0), 0.0, 4
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    area = float(mask.mean())
    box_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    fill = float(mask.sum() / box_area)
    border = int(bbox[0] <= 2) + int(bbox[1] <= 2) + int(bbox[2] >= mask.shape[1] - 2) + int(bbox[3] >= mask.shape[0] - 2)
    return area, bbox, (float(xs.mean()), float(ys.mean())), fill, border


def sam_candidates(processor, model, image: Image.Image, x_center: float) -> list[Candidate]:
    candidates: list[Candidate] = []
    for point in salience_points(image, x_center):
        inputs = processor(image, input_points=[[[point[0], point[1]]]], return_tensors="pt")
        model_inputs = {
            key: value.to("cuda")
            for key, value in inputs.items()
            if key not in ("original_sizes", "reshaped_input_sizes")
        }
        with torch.inference_mode():
            outputs = model(**model_inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]
        ious = outputs.iou_scores[0, 0].detach().cpu()
        for candidate_index in range(masks.shape[1]):
            raw = masks[0, candidate_index].numpy().astype(bool)
            component = seeded_component(raw, point)
            area, bbox, centroid, compact, border = mask_geometry(component)
            if not (0.008 <= area <= 0.42):
                continue
            cx = centroid[0] / SIZE
            side_distance = abs(cx - x_center)
            area_preference = abs(area - 0.16)
            span_x = (bbox[2] - bbox[0]) / SIZE
            span_y = (bbox[3] - bbox[1]) / SIZE
            score = (
                float(ious[candidate_index])
                + 0.55 * compact
                + 0.25 * span_x
                + 0.85 * span_y
                - 2.5 * side_distance
                - 0.8 * area_preference
                - 0.9 * border
            )
            candidates.append(Candidate(component, point, float(ious[candidate_index]), score, area, bbox, centroid))
    candidates.sort(key=lambda item: item.score, reverse=True)
    unique: list[Candidate] = []
    for candidate in candidates:
        if all(float((candidate.mask & old.mask).sum()) / max(1, float((candidate.mask | old.mask).sum())) < 0.97 for old in unique):
            unique.append(candidate)
        if len(unique) >= 12:
            break
    if not unique:
        raise RuntimeError("SAM produced no geometrically plausible instance candidate")
    return unique


def choose_joint(candidate_sets: list[list[Candidate]]) -> list[Candidate]:
    if len(candidate_sets) > 4:
        raise ValueError("The exhaustive pilot selector supports at most four entities")
    best = None
    best_value = -1e30
    for combination in itertools.product(*candidate_sets):
        overlap = 0.0
        for a, b in itertools.combinations(combination, 2):
            overlap += float((a.mask & b.mask).mean())
        value = sum(item.score for item in combination) - 30.0 * overlap
        if value > best_value:
            best_value = value
            best = combination
    assert best is not None
    return list(best)


def clean_and_partition(chosen: list[Candidate]) -> tuple[list[np.ndarray], np.ndarray, dict]:
    masks = []
    for candidate in chosen:
        filled = fill_holes(candidate.mask)
        masks.append(filled)
    raw_overlap = np.stack(masks).sum(axis=0) > 1
    centroids = [mask_geometry(mask)[2] for mask in masks]
    owners = [np.zeros_like(masks[0], dtype=bool) for _ in masks]
    stack = np.stack(masks)
    count = stack.sum(axis=0)
    for index in range(len(masks)):
        owners[index] |= masks[index] & (count == 1)
    ys, xs = np.where(count > 1)
    for y, x in zip(ys.tolist(), xs.tolist()):
        owner = int(np.argmin([(x - cx) ** 2 + (y - cy) ** 2 for cx, cy in centroids]))
        owners[owner][y, x] = True
    background = ~np.logical_or.reduce(owners)
    partition = background.astype(np.uint8)
    for owner in owners:
        partition += owner.astype(np.uint8)
    diagnostics = {
        "raw_overlap_fraction": float(raw_overlap.mean()),
        "owner_fractions_image": [float(owner.mean()) for owner in owners],
        "background_fraction_image": float(background.mean()),
        "partition_min_image": int(partition.min()),
        "partition_max_image": int(partition.max()),
    }
    if diagnostics["raw_overlap_fraction"] > 0.03:
        raise RuntimeError(f"Instance masks overlap too much: {diagnostics['raw_overlap_fraction']:.4f}")
    if diagnostics["partition_min_image"] != 1 or diagnostics["partition_max_image"] != 1:
        raise RuntimeError("Image-space ownership is not one-hot")
    return owners, background, diagnostics


def save_segmentation(image: Image.Image, owners: list[np.ndarray], out: Path, entities: list[dict], diagnostics: dict):
    out.mkdir(parents=True, exist_ok=True)
    colors = [(58, 134, 203), (210, 82, 60), (84, 158, 109), (139, 92, 184)]
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for index, owner in enumerate(owners):
        Image.fromarray((owner * 255).astype(np.uint8), mode="L").save(out / f"owner_{index + 1}.png")
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        overlay[owner] = 0.58 * overlay[owner] + 0.42 * color
    canvas = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for index, owner in enumerate(owners):
        area, bbox, centroid, _, _ = mask_geometry(owner)
        draw.rectangle(bbox, outline=colors[index % len(colors)], width=5)
        draw.text((bbox[0] + 8, bbox[1] + 8), entities[index]["name"], fill=colors[index % len(colors)])
        diagnostics.setdefault("entities", []).append({
            "name": entities[index]["name"], "area": area, "bbox": bbox, "centroid": centroid
        })
    canvas.save(out / "overlay.png")
    (out / "segmentation.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")


def image_masks_to_latent(owners: list[np.ndarray], latent: torch.Tensor):
    tensors = []
    for owner in owners:
        value = torch.from_numpy(owner.astype(np.float32))[None, None].to(latent.device)
        value = F.interpolate(value, size=latent.shape[-2:], mode="nearest")
        tensors.append(value)
    count = torch.stack(tensors).sum(0)
    # Nearest resize preserves one-hot pixels unless multiple image masks collapse into the same latent cell.
    if float((count > 1).float().mean()) > 0:
        centers = []
        for owner in owners:
            _, _, centroid, _, _ = mask_geometry(owner)
            centers.append((centroid[0] / 8.0, centroid[1] / 8.0))
        ys, xs = torch.where(count[0, 0] > 1)
        for y, x in zip(ys.tolist(), xs.tolist()):
            winner = min(range(len(centers)), key=lambda i: (x - centers[i][0]) ** 2 + (y - centers[i][1]) ** 2)
            for i in range(len(tensors)):
                tensors[i][0, 0, y, x] = 1.0 if i == winner else 0.0
    background = 1.0 - torch.stack(tensors).sum(0).clamp(0, 1)
    partition = background + torch.stack(tensors).sum(0)
    if float(partition.min()) != 1.0 or float(partition.max()) != 1.0:
        raise RuntimeError("Latent ownership is not exactly one-hot")
    return tensors, background


def owner_bounds(owner: torch.Tensor, patch_size: int, margin: int = 4):
    ys, xs = torch.where(owner[0, 0] > 0.5)
    if not len(xs):
        raise RuntimeError("Empty latent owner")
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(owner.shape[-1], int(xs.max()) + 1 + margin)
    bottom = min(owner.shape[-2], int(ys.max()) + 1 + margin)
    left = left // patch_size * patch_size
    top = top // patch_size * patch_size
    right = min(owner.shape[-1], math.ceil(right / patch_size) * patch_size)
    bottom = min(owner.shape[-2], math.ceil(bottom / patch_size) * patch_size)
    return left, top, right, bottom


def cfg_full(pipe, latent, timestep, encoded, index, transformer_pair):
    embeds = torch.cat([encoded["negative"], encoded["positive"][index:index + 1]], dim=0)
    pooled = torch.cat([encoded["negative_pooled"], encoded["positive_pooled"][index:index + 1]], dim=0)
    uncond, cond = transformer_pair(pipe, latent, timestep, embeds, pooled).chunk(2)
    return uncond + GUIDANCE * (cond - uncond)


def native_ss_trajectory(pipe, initial, encoded, helper_functions):
    prepare_schedule, transformer_pair, _ = helper_functions
    latent = initial.clone()
    timesteps, mu = prepare_schedule(pipe, latent, STEPS)
    scheduler = copy.deepcopy(pipe.scheduler)
    states = []
    with torch.inference_mode():
        for timestep in timesteps:
            velocity = cfg_full(pipe, latent, timestep, encoded, 0, transformer_pair)
            latent = scheduler.step(
                velocity.to(latent.dtype), timestep, latent, return_dict=False
            )[0].to(initial.dtype)
            states.append(latent.clone())
    return states, timesteps, mu


def denoise_specialists(pipe, initial, encoded, owners, background, native_states, helper_functions):
    prepare_schedule, transformer_pair, region_cfg_score = helper_functions
    latent = initial.clone()
    timesteps, mu = prepare_schedule(pipe, latent, STEPS)
    patch_size = int(pipe.transformer.config.patch_size)
    bounds = [owner_bounds(owner, patch_size) for owner in owners]
    local_schedulers = [copy.deepcopy(pipe.scheduler) for _ in owners]
    audit = []
    with torch.inference_mode():
        for step_index, timestep in enumerate(timesteps):
            # Expert N+1 is the frozen native-SS scaffold trajectory. It owns only the
            # complement of all entity masks and therefore preserves the original scene.
            next_latent = background * native_states[step_index]
            committed_deltas = []
            for entity_index, (owner, bounds_i, scheduler) in enumerate(
                zip(owners, bounds, local_schedulers), start=1
            ):
                left, top, right, bottom = bounds_i
                crop = latent[:, :, top:bottom, left:right]
                score = region_cfg_score(
                    pipe, crop, timestep,
                    encoded["negative"], encoded["positive"][entity_index:entity_index + 1],
                    encoded["negative_pooled"], encoded["positive_pooled"][entity_index:entity_index + 1],
                    GUIDANCE,
                )
                candidate = scheduler.step(
                    score.to(crop.dtype), timestep, crop, return_dict=False
                )[0].to(initial.dtype)
                owner_crop = owner[:, :, top:bottom, left:right]
                next_latent[:, :, top:bottom, left:right] += owner_crop * candidate
                delta = torch.zeros_like(latent)
                delta[:, :, top:bottom, left:right] = owner_crop * (candidate - crop)
                committed_deltas.append(delta)
            outside = []
            for owner, delta in zip(owners, committed_deltas):
                outside.append(float(((1.0 - owner) * delta).square().mean().sqrt().item()))
            latent = next_latent.to(initial.dtype)
            partition = background + torch.stack(owners).sum(0)
            audit.append({
                "step": step_index,
                "timestep": float(timestep.item()),
                "partition_min": float(partition.min().item()),
                "partition_max": float(partition.max().item()),
                "outside_write_rms": outside,
                "background_native_match_rms": float(
                    (background * (latent - native_states[step_index])).square().mean().sqrt().item()
                ),
                "state_rms": float(latent.square().mean().sqrt().item()),
            })
    return latent, {"mu": mu, "bounds": bounds, "step_audit": audit}


def make_comparison(ss: Image.Image, overlay: Image.Image, lsda: Image.Image, out: Path):
    width = 3 * SIZE
    canvas = Image.new("RGB", (width, SIZE + 52), (250, 246, 238))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate((("Native SS", ss), ("SS ownership", overlay), ("LSDA", lsda))):
        canvas.paste(image.resize((SIZE, SIZE), Image.Resampling.LANCZOS), (index * SIZE, 52))
        draw.text((index * SIZE + 16, 15), label, fill=(31, 78, 121))
    canvas.save(out, quality=95, subsampling=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-ss", action="store_true")
    parser.add_argument("--reuse-segmentation", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_dir = ROOT / "runs" / config["run_id"]
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    sys.path.insert(0, str(HELPERS))
    import phase1_common
    phase1_common.MODEL_DIR = MODEL
    from diffusers import StableDiffusion3Pipeline
    from phase212_regional_score import prepare_schedule, transformer_pair
    from phase213_multidiffusion_crop import region_cfg_score
    from phase221_group_overlap_arbitration import encode_prompts

    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL, torch_dtype=torch.float16, local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    seed = int(config["seed"])
    started = time.time()
    initial, latent_sha = make_latent(pipe, seed)
    prompts = (config["ss_prompt"], *[entity["prompt"] for entity in config["entities"]])
    encoded = encode_prompts(pipe, prompts)
    helpers = (prepare_schedule, transformer_pair, region_cfg_score)
    native_states, _, native_mu = native_ss_trajectory(pipe, initial, encoded, helpers)
    ss = decode(pipe, native_states[-1])
    ss_path = run_dir / "native_ss.png"
    ss.save(ss_path)

    segmentation_dir = run_dir / "segmentation"
    if args.reuse_segmentation and all(
        (segmentation_dir / f"owner_{index + 1}.png").exists()
        for index in range(len(config["entities"]))
    ):
        owners_image = [
            np.asarray(Image.open(segmentation_dir / f"owner_{index + 1}.png").convert("L")) > 127
            for index in range(len(config["entities"]))
        ]
        segmentation_diagnostics = json.loads(
            (segmentation_dir / "segmentation.json").read_text(encoding="utf-8")
        )
    else:
        sam_processor = SamProcessor.from_pretrained(SAM_MODEL, local_files_only=True)
        sam_model = SamModel.from_pretrained(SAM_MODEL, local_files_only=True).to("cuda").eval()
        candidate_sets = []
        for index, entity in enumerate(config["entities"]):
            x = expected_x(entity.get("position", ""), index, len(config["entities"]))
            candidate_sets.append(sam_candidates(sam_processor, sam_model, ss, x))
        chosen = choose_joint(candidate_sets)
        owners_image, _, segmentation_diagnostics = clean_and_partition(chosen)
        segmentation_diagnostics["chosen"] = [
            {"point": item.point, "sam_iou": item.sam_iou, "selection_score": item.score,
             "candidate_area": item.area, "bbox": item.bbox}
            for item in chosen
        ]
        save_segmentation(ss, owners_image, segmentation_dir, config["entities"], segmentation_diagnostics)
        del sam_model, sam_processor
        torch.cuda.empty_cache()

    owners, background = image_masks_to_latent(owners_image, initial)
    final_latent, denoise_diagnostics = denoise_specialists(
        pipe, initial, encoded, owners, background, native_states, helpers,
    )
    lsda = decode(pipe, final_latent)
    lsda.save(run_dir / "lsda.png")
    overlay = Image.open(run_dir / "segmentation" / "overlay.png").convert("RGB")
    make_comparison(ss, overlay, lsda, run_dir / "comparison.jpg")
    record = {
        "run_id": config["run_id"], "seed": seed, "latent_sha256": latent_sha,
        "steps": STEPS, "guidance": GUIDANCE,
        "specialist_count": len(config["entities"]) + 1,
        "entity_prompts": [entity["prompt"] for entity in config["entities"]],
        "background_expert": "same-seed native SS trajectory, mask complement only",
        "native_mu": native_mu,
        "external_visual_information": "native SS contour only",
        "standalone_sl_ll_image_or_latent_used": False,
        "segmentation": segmentation_diagnostics,
        "denoising": denoise_diagnostics,
        "seconds": time.time() - started,
    }
    (run_dir / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "run_dir": str(run_dir), "seconds": record["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
