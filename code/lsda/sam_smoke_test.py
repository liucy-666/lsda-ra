from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from transformers import SamModel, SamProcessor

from lsda_pipeline import (
    choose_joint,
    clean_and_partition,
    sam_candidates,
    save_segmentation,
)


DEFAULT_PROMPT = (
    "On a neutral studio background, a Chinese blue-and-white porcelain plate "
    "is placed on the left and a Japanese Kutani porcelain vase is placed on "
    "the right. Both objects are fully visible, spatially separated, and similar "
    "in apparent size."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--sam-model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=False)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=4.5,
        height=1024,
        width=1024,
        generator=generator,
    ).images[0]
    native_path = args.output_dir / "native_ss_seed999.png"
    image.save(native_path)

    del pipe
    torch.cuda.empty_cache()

    processor = SamProcessor.from_pretrained(
        args.sam_model_dir,
        local_files_only=True,
    )
    model = SamModel.from_pretrained(
        args.sam_model_dir,
        local_files_only=True,
    ).to("cuda")
    model.eval()

    candidate_sets = [
        sam_candidates(processor, model, image, 0.28),
        sam_candidates(processor, model, image, 0.73),
    ]
    chosen = choose_joint(candidate_sets)
    owners, _background, diagnostics = clean_and_partition(chosen)
    entities = [
        {"name": "A: Chinese blue-and-white plate"},
        {"name": "B: Japanese Kutani vase"},
    ]
    segmentation_dir = args.output_dir / "segmentation"
    save_segmentation(image, owners, segmentation_dir, entities, diagnostics)

    manifest = {
        "prompt": args.prompt,
        "seed": args.seed,
        "model_dir": str(args.model_dir),
        "sam_model_dir": str(args.sam_model_dir),
        "sam_variant": "sam-vit-base",
        "elapsed_seconds": time.time() - started,
        "outputs": {
            "native_ss": str(native_path),
            "overlay": str(segmentation_dir / "overlay.png"),
            "owner_a": str(segmentation_dir / "owner_1.png"),
            "owner_b": str(segmentation_dir / "owner_2.png"),
        },
        "segmentation": diagnostics,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
