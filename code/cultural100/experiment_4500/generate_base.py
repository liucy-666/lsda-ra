from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import PngImagePlugin
from diffusers import StableDiffusion3Pipeline


ROOT = Path("/science/wx/pry/AAA_Experiment/cultural100_v1")
MANIFEST = ROOT / "base_manifest.json"
MODEL = Path("/science/wx/pry/models/stable-diffusion-3.5-large")
STEPS = 28
GUIDANCE = 4.5
SIZE = 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("nshards", type=int)
    args = parser.parse_args()

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks = [row for i, row in enumerate(payload["tasks"]) if i % args.nshards == args.shard]
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for row in tasks:
        image_dir = ROOT / "base" / "images" / row["pair_id"] / row["image_type"]
        sidecar_dir = ROOT / "base" / "sidecars" / row["pair_id"] / row["image_type"]
        image_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f'{row["task_id"]}.png'
        sidecar_path = sidecar_dir / f'{row["task_id"]}.json'
        if image_path.exists() and sidecar_path.exists():
            print(json.dumps({"event": "skip", "task_id": row["task_id"]}), flush=True)
            continue

        generator = torch.Generator(device="cuda").manual_seed(int(row["latent_seed"]))
        started = time.time()
        with torch.inference_mode():
            image = pipe(
                prompt=row["prompt"],
                prompt_2=row["prompt"],
                prompt_3=row["prompt"],
                negative_prompt="",
                negative_prompt_2="",
                negative_prompt_3="",
                num_inference_steps=STEPS,
                guidance_scale=GUIDANCE,
                height=SIZE,
                width=SIZE,
                generator=generator,
            ).images[0]
        seconds = time.time() - started
        record = {
            **row,
            "method": "native_sd35",
            "steps": STEPS,
            "guidance_scale": GUIDANCE,
            "resolution": [SIZE, SIZE],
            "seconds": seconds,
            "shard": args.shard,
        }
        info = PngImagePlugin.PngInfo()
        info.add_text("task_id", row["task_id"])
        info.add_text("prompt", row["prompt"])
        info.add_text("latent_seed", str(row["latent_seed"]))
        info.add_text("metadata", json.dumps(record, ensure_ascii=False, sort_keys=True))
        image.save(image_path, pnginfo=info)
        sidecar_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        print(json.dumps({"event": "done", "task_id": row["task_id"], "seconds": seconds,
                          "sha256": digest}), flush=True)
        del image
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
