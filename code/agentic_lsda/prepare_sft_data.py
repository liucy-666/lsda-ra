"""Prepare Stage-1 SFT training data from the local 900-task experiment.

Assembles one JSONL record per (sample_id x condition) with:
  - image path (native_SS or lsda_clean single image) + standalone A/B paths
  - prompts (global SS prompt + entity A/B prompts) from lsda_manifest.json
  - Qwen/Gemini binding verdicts from binary_vqa_v2 ratings (per eval_id)
  - transition class + derived SFT task labels:
      task1_drift_class      : image-level drift classification target
      task2_leakage_direction: A_to_B / B_to_A / bidirectional / none
      task3_accept_reject    : accept / reject / no_op (lsda_clean condition only)

Usage:
  python prepare_sft_data.py [--out <sft_training_manifest.jsonl>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP = PROJECT_ROOT / "experiment" / "2026_8_25_EXP_1" / "cultural100_records" / "experiment_4500"
DATA = PROJECT_ROOT / "data"

DEFAULTS = {
    "manifest": EXP / "lsda_manifest.json",
    "transitions": EXP / "binary_vqa_v2" / "source_data" / "qwen_paired_transitions.csv",
    "ratings_root": EXP / "binary_vqa_v2" / "ratings",
    "image_ss": DATA / "SS" / "2026_8_25_EXP_1",
    "image_lsda": DATA / "LSDA" / "2026_8_25_EXP_1",
    "image_sa": DATA / "Standalone_A" / "2026_8_25_EXP_1",
    "image_sb": DATA / "Standalone_B" / "2026_8_25_EXP_1",
    "out": PROJECT_ROOT / "experiment" / "2026_9_1_EXP_1" / "manifests" / "sft_training_manifest.jsonl",
}


def load_ratings(ratings_root: Path) -> dict[str, dict]:
    """rater -> eval_id -> verdict row (last occurrence wins)."""
    result: dict[str, dict[str, dict]] = {"QWEN": {}, "GEMINI": {}}
    for rater in ("QWEN", "GEMINI"):
        folder = ratings_root / rater
        if not folder.exists():
            continue
        for path in sorted(folder.glob("ratings_*.jsonl")):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("eval_id"):
                    result[rater][row["eval_id"]] = row
    return result


def leakage_direction(row: dict | None) -> str:
    if not row:
        return "unknown"
    a2b = bool(row.get("A_to_B_leakage"))
    b2a = bool(row.get("B_to_A_leakage"))
    if a2b and b2a:
        return "bidirectional"
    if a2b:
        return "A_to_B"
    if b2a:
        return "B_to_A"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key}", type=Path, default=default)
    parser.add_argument("--img-format", type=str, default="jpg")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prompts_by_native = {
        task["native_task_id"]: task for task in manifest["tasks"]
    }
    ratings = load_ratings(args.ratings_root)

    rows = list(csv.DictReader(args.transitions.open(encoding="utf-8-sig")))
    print(f"transitions rows : {len(rows)}")
    print(f"manifest tasks   : {len(manifest['tasks'])}")
    print(
        "ratings coverage  : QWEN={} GEMINI={}".format(
            len(ratings["QWEN"]), len(ratings["GEMINI"])
        )
    )

    records: list[dict] = []
    missing = {"prompt": 0, "image": 0, "rating_q": 0, "rating_g": 0}
    for row in rows:
        sample = row["sample_id"]
        native_eval, lsda_eval = row["native_eval_id"], row["lsda_eval_id"]
        prompt_task = prompts_by_native.get(row.get("native_task_id", ""))
        if prompt_task is None:
            # fall back: search by native_task_id built from fields if CSV lacks it
            prompt_task = next(
                (
                    t for t in manifest["tasks"]
                    if t["pair_id"] == row["pair_id"]
                    and t["seed_group"] == int(row["seed_group"])
                    and t["replicate"] == int(row["replicate"])
                ),
                None,
            )
        if prompt_task is None:
            missing["prompt"] += 1
            global_prompt = entity_a = entity_b = None
        else:
            global_prompt = prompt_task["global_prompt"]
            entity_a = prompt_task["entity_A_prompt"]
            entity_b = prompt_task["entity_B_prompt"]

        for condition, eval_id, drift_key, correct_key, img_root in (
            ("native_SS", native_eval, "native_drift", "native_correct", args.image_ss),
            ("lsda_clean", lsda_eval, "lsda_drift", "lsda_correct", args.image_lsda),
        ):
            image = img_root / f"{sample}.{args.img_format}"
            if not image.exists():
                missing["image"] += 1
            q, g = ratings["QWEN"].get(eval_id), ratings["GEMINI"].get(eval_id)
            if q is None:
                missing["rating_q"] += 1
            if g is None:
                missing["rating_g"] += 1
            vlm = {}
            for rater, verdict in (("QWEN", q), ("GEMINI", g)):
                if verdict:
                    vlm[rater] = {
                        "correct_binding": verdict.get("correct_binding"),
                        "A_to_B_leakage": verdict.get("A_to_B_leakage"),
                        "B_to_A_leakage": verdict.get("B_to_A_leakage"),
                        "left_choice": verdict.get("left_choice"),
                        "right_choice": verdict.get("right_choice"),
                    }
            binding_agree = None
            if q and g:
                binding_agree = bool(q.get("correct_binding")) == bool(g.get("correct_binding"))

            transition = row["transition"]
            if condition == "lsda_clean":
                accept = {
                    "restored": "accept",
                    "worsened": "reject",
                    "persistent_failure": "reject",
                    "both_correct": "no_op",
                }.get(transition, "unknown")
            else:
                accept = None

            # leakage direction: use Qwen as primary, note Gemini disagreement
            leak_q = leakage_direction(q)
            leak_g = leakage_direction(g)

            records.append(
                {
                    "sample_id": sample,
                    "pair_id": row["pair_id"],
                    "pair_index": int(row["pair_index"]) if row["pair_index"] else None,
                    "seed_group": int(row["seed_group"]) if row["seed_group"] else None,
                    "replicate": int(row["replicate"]) if row["replicate"] else None,
                    "latent_seed": int(row["latent_seed"]) if row["latent_seed"] else None,
                    "condition": condition,
                    "eval_id": eval_id,
                    "image": image.as_posix(),
                    "standalone_A_image": (args.image_sa / f"{sample}.{args.img_format}").as_posix(),
                    "standalone_B_image": (args.image_sb / f"{sample}.{args.img_format}").as_posix(),
                    "global_prompt": global_prompt,
                    "entity_A_prompt": entity_a,
                    "entity_B_prompt": entity_b,
                    "transition": transition,
                    "drift_label": int(row[drift_key]) if row.get(drift_key) in ("0", "1") else None,
                    "correct_label": int(row[correct_key]) if row.get(correct_key) in ("0", "1") else None,
                    "vlm": vlm,
                    "vlm_binding_agreement": binding_agree,
                    "task1_drift_class": int(row[drift_key]) if row.get(drift_key) in ("0", "1") else None,
                    "task2_leakage_direction_qwen": leak_q,
                    "task2_leakage_direction_gemini": leak_g,
                    "task3_accept_reject": accept,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    from collections import Counter
    transitions = Counter(r["transition"] for r in records)
    native_drift = sum(1 for r in records if r["condition"] == "native_SS" and r["drift_label"] == 1)
    lsda_drift = sum(1 for r in records if r["condition"] == "lsda_clean" and r["drift_label"] == 1)
    agree = [r["vlm_binding_agreement"] for r in records if r["vlm_binding_agreement"] is not None]
    print("-" * 72)
    print(f"records written  : {len(records)} -> {args.out}")
    print(f"missing          : {missing}")
    print(f"transition dist  : {dict(transitions)}")
    print(
        f"Qwen drift rate  : native={native_drift}/900 ({native_drift/9:.1f}%)  "
        f"lsda={lsda_drift}/900 ({lsda_drift/9:.1f}%)"
    )
    if agree:
        print(
            f"VLM binding agreement (native+lsda): {sum(agree)}/{len(agree)} "
            f"({100*sum(agree)/len(agree):.1f}%)"
        )


if __name__ == "__main__":
    main()
