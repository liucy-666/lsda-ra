from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import PngImagePlugin


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MANIFEST = ROOT / "lsda_manifest.json"
MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
SRC = Path("/science/wx/pry/baseline_v2/lsda_v2_src")
STEPS = 28
HANDOFF = 7
GUIDANCE = 4.5
BACKGROUND = "neutral studio background and surface only, without objects"


@dataclass(frozen=True)
class Scenario:
    name: str
    global_prompt: str
    appearance_prompts: tuple[str, str]
    lowpass_alpha: tuple[float, float] = (0.0, 0.0)


def make_latent(pipe, seed: int):
    generator = torch.Generator("cuda").manual_seed(seed)
    latent = pipe.prepare_latents(
        1, pipe.transformer.config.in_channels, 1024, 1024,
        torch.float16, torch.device("cuda"), generator, None,
    )
    digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    return latent, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("nshards", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(SRC))
    import phase1_common
    phase1_common.MODEL_DIR = MODEL
    from diffusers import StableDiffusion3Pipeline
    from phase221_group_overlap_arbitration import encode_prompts
    from phase225_irreversible_handoff import denoise as hard_handoff_denoise

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = [row for i, row in enumerate(payload["tasks"]) if i % args.nshards == args.shard]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for row in tasks:
        image_dir = ROOT / "lsda_original" / "images" / row["pair_id"]
        sidecar_dir = ROOT / "lsda_original" / "sidecars" / row["pair_id"]
        image_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        stem = row["task_id"].replace("lsda_ra", "lsda_original")
        image_path = image_dir / f"{stem}.png"
        sidecar_path = sidecar_dir / f"{stem}.json"
        if image_path.exists() and sidecar_path.exists():
            print(json.dumps({"event": "skip", "task_id": stem}), flush=True)
            continue

        prompts = (row["global_prompt"], row["entity_A_prompt"], row["entity_B_prompt"], BACKGROUND)
        encoded = encode_prompts(pipe, prompts)
        initial, latent_sha = make_latent(pipe, int(row["latent_seed"]))
        scenario = Scenario(stem, row["global_prompt"],
                            (row["entity_A_prompt"], row["entity_B_prompt"]))
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.inference_mode():
            image, diagnostics, owners, conflict, background_owner = hard_handoff_denoise(
                pipe, initial, encoded, scenario,
                ROOT / "lsda_original" / "masks" / row["task_id"],
            )
        record = {
            **row,
            "task_id": stem,
            "method": "original LSDA Phase 2.25 irreversible one-hot handoff",
            "region_source": row["native_task_id"],
            "region_source_type": "native_SS self-segmentation",
            "standalone_visual_information_used": False,
            "sl_ll_information_used": False,
            "background_prompt": BACKGROUND,
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "handoff_step": HANDOFF,
            "latent_sha256": latent_sha,
            "seconds": time.time() - started,
            "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "diagnostics": diagnostics,
        }
        info = PngImagePlugin.PngInfo()
        info.add_text("task_id", stem)
        info.add_text("prompt", row["global_prompt"])
        info.add_text("latent_seed", str(row["latent_seed"]))
        info.add_text("metadata", json.dumps(record, ensure_ascii=False, sort_keys=True))
        image.save(image_path, pnginfo=info)
        sidecar_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "done", "task_id": stem,
                          "seconds": record["seconds"]}), flush=True)
        del encoded, initial, image, owners, conflict, background_owner
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
