"""Batch LSDA v1.5 driver with two independent commit variants.

variant "mask"  : commit SAM masks (dilated) + feather  -> seam follows the object contour, feathered.
variant "alpha" : commit padded rectangles + feather    -> expert background cross-fades into native.
                  (alpha with --feather-px 0 reproduces the v1.4 hard-rect baseline)

Native SS is decoded from the same-seed trajectory (1024) and used only for SAM segmentation,
so no external SS file is required.

Jobs JSON entries: {run_id, pair, seed, a_prompt, b_prompt, ss_prompt, kind}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "helpers"))

import lsda_pipeline_v15 as L  # noqa: E402
from phase212_regional_score import prepare_schedule, transformer_pair  # noqa: E402
from phase213_multidiffusion_crop import region_cfg_score  # noqa: E402
from phase221_group_overlap_arbitration import encode_prompts  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--sam-model-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["mask", "alpha"], required=True)
    ap.add_argument("--rect-pad", type=int, default=16)
    ap.add_argument("--dilate-px", type=int, default=0)
    ap.add_argument("--feather-px", type=int, default=12)
    args = ap.parse_args()

    L.RECT_PAD = args.rect_pad
    L.DILATE_PX = args.dilate_px
    L.FEATHER_PX = args.feather_px

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    helpers = (prepare_schedule, transformer_pair, region_cfg_score)

    sam_processor = SamProcessor.from_pretrained(args.sam_model_dir, local_files_only=True)
    sam_model = SamModel.from_pretrained(args.sam_model_dir, local_files_only=True).to("cuda").eval()

    done_count = 0
    for job in jobs:
        run_dir = args.out / "runs" / job["run_id"]
        out_png = run_dir / "lsda.png"
        if out_png.exists():
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        try:
            seed = int(job["seed"])
            initial, latent_sha = L.make_latent(pipe, seed)
            encoded = encode_prompts(pipe, (job["ss_prompt"], job["a_prompt"], job["b_prompt"]))
            native_states, _, native_mu = L.native_ss_trajectory(pipe, initial, encoded, helpers)
            native = L.decode(pipe, native_states[-1])
            native.save(run_dir / "native_ss.png")

            xs = [L.expected_x("left", 0, 2), L.expected_x("right", 1, 2)]
            candidate_sets = [L.sam_candidates(sam_processor, sam_model, native, x) for x in xs]
            chosen = L.choose_joint(candidate_sets)
            owners_image, _, seg_diag = L.clean_and_partition(chosen)
            commit = L.build_commit_masks(owners_image, args.variant, args.rect_pad)

            owners, background = L.image_masks_to_latent(commit, initial)
            final_latent, denoise_diag = L.denoise_specialists(
                pipe, initial, encoded, owners, background, native_states, helpers
            )
            image = L.decode(pipe, final_latent)
            image.save(out_png)
            (run_dir / "audit.json").write_text(
                json.dumps(
                    {
                        "run_id": job["run_id"], "kind": job["kind"], "pair": job["pair"],
                        "seed": seed, "variant": args.variant,
                        "rect_pad": args.rect_pad, "dilate_px": args.dilate_px, "feather_px": args.feather_px,
                        "latent_sha256": latent_sha,
                        "entity_prompts": [job["a_prompt"], job["b_prompt"]],
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
            with (args.out / "logs" / "failures.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"run_id": job["run_id"], "error": type(exc).__name__, "msg": str(exc)}) + "\n")
            print(json.dumps({"failed": job["run_id"], "error": str(exc)[:120]}), flush=True)

    print(json.dumps({"event": "complete", "jobs": len(jobs), "done": done_count}), flush=True)


if __name__ == "__main__":
    main()
