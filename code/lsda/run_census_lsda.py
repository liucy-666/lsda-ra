"""Batch LSDA repair driver for the KA/ME census (v1.4 rectangle regions, SAM locator).

Jobs JSON entries: {run_id, pair, seed, a_prompt, b_prompt, ss_prompt, kind}.
For each job: native SS -> SAM tight masks (cached per pair/seed) -> rectangles -> LSDA.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "helpers"))

import lsda_pipeline_v14_rect as L  # noqa: E402
from phase212_regional_score import prepare_schedule, transformer_pair  # noqa: E402
from phase213_multidiffusion_crop import region_cfg_score  # noqa: E402
from phase221_group_overlap_arbitration import encode_prompts  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--sam-model-dir", type=Path, required=True)
    ap.add_argument("--rect-pad", type=int, default=16)
    ap.add_argument("--dilate-px", type=int, default=0)
    args = ap.parse_args()

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (prepare_schedule, transformer_pair, region_cfg_score)

    sam_processor = SamProcessor.from_pretrained(args.sam_model_dir, local_files_only=True)
    sam_model = SamModel.from_pretrained(args.sam_model_dir, local_files_only=True).to("cuda").eval()

    mask_cache: dict = {}
    done_count = 0
    for job in jobs:
        run_dir = args.out / "runs" / job["run_id"]
        out_png = run_dir / "lsda.png"
        if out_png.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            native = Image.open(args.images_dir / f"pair{job['pair']:03d}_seed{job['seed']}_SS.png").convert("RGB")
            key = (job["pair"], job["seed"])
            if key not in mask_cache:
                xs = [L.expected_x("left", 0, 2), L.expected_x("right", 1, 2)]
                candidate_sets = [L.sam_candidates(sam_processor, sam_model, native, x) for x in xs]
                chosen = L.choose_joint(candidate_sets)
                owners_image, _, seg_diag = L.clean_and_partition(chosen)
                mask_cache[key] = (owners_image, seg_diag)
            owners_image, seg_diag = mask_cache[key]
            rects = L.rectangularize_owners(owners_image, args.rect_pad)

            initial, latent_sha = L.make_latent(pipe, int(job["seed"]))
            encoded = encode_prompts(pipe, (job["ss_prompt"], job["a_prompt"], job["b_prompt"]))
            native_states, _, native_mu = L.native_ss_trajectory(pipe, initial, encoded, helpers)
            owners, background = L.image_masks_to_latent(rects, initial)
            final_latent, denoise_diag = L.denoise_specialists(
                pipe, initial, encoded, owners, background, native_states, helpers
            )
            image = L.decode(pipe, final_latent)
            image.save(out_png)
            (run_dir / "audit.json").write_text(
                json.dumps(
                    {
                        "run_id": job["run_id"], "kind": job["kind"], "pair": job["pair"],
                        "seed": job["seed"], "latent_sha256": latent_sha,
                        "entity_prompts": [job["a_prompt"], job["b_prompt"]],
                        "rect_pad": args.rect_pad, "dilate_px": args.dilate_px,
                        "segmentation": {k: v for k, v in seg_diag.items() if k != "chosen"},
                        "native_mu": native_mu,
                        "seconds": round(time.time() - started, 1),
                    },
                    ensure_ascii=False, indent=1,
                ),
                encoding="utf-8",
            )
            done_count += 1
            print(json.dumps({"done": job["run_id"], "seconds": round(time.time() - started, 1)}), flush=True)
        except Exception as exc:  # noqa: BLE001
            with (log_dir / "failures.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run_id": job["run_id"], "error": type(exc).__name__, "msg": str(exc)}) + "\n")
            print(json.dumps({"failed": job["run_id"], "error": str(exc)[:120]}), flush=True)

    print(json.dumps({"event": "complete", "jobs": len(jobs), "done": done_count}), flush=True)


if __name__ == "__main__":
    main()
