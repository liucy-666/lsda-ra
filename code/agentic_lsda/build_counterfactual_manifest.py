from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from schema import SCHEMA_VERSION, ReplayAction, ReplayTask, task_to_dict, write_manifest


DEFAULT_STRATA = {
    "restored": 5,
    "persistent_failure": 3,
    "worsened": 2,
    "both_correct": 2,
}


def parse_steps(value: str, num_steps: int) -> list[int]:
    steps = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not steps:
        raise ValueError("at least one intervention step is required")
    invalid = [step for step in steps if not 0 <= step < num_steps]
    if invalid:
        raise ValueError(f"steps outside [0, {num_steps - 1}]: {invalid}")
    return steps


def parse_strata(value: str) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_STRATA)
    result: dict[str, int] = {}
    for part in value.split(","):
        name, count = part.split("=", 1)
        result[name.strip()] = int(count)
    if not result or any(count < 0 for count in result.values()):
        raise ValueError("strata must contain non-negative counts")
    return result


def load_source_tasks(path: Path) -> dict[tuple[str, int, int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks", [])
    index = {
        (row["pair_id"], int(row["seed_group"]), int(row["replicate"])): row
        for row in tasks
    }
    if len(index) != len(tasks):
        raise ValueError("source manifest contains duplicate pair/seed/replicate keys")
    return index


def load_transitions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "pair_id", "seed_group", "replicate", "transition"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"transition CSV lacks required columns: {sorted(required)}")
    return rows


def valid_mask_samples(root: Path) -> tuple[set[str], dict[str, str]]:
    valid: set[str] = set()
    rejected: dict[str, str] = {}
    for path in sorted(root.glob("*_mask.npz")):
        sample_id = path.name.removesuffix("_mask.npz")
        try:
            with np.load(path) as payload:
                if not {"left", "right"}.issubset(payload.files):
                    raise ValueError("missing left/right")
                left = payload["left"].astype(bool)
                right = payload["right"].astype(bool)
            if left.shape != right.shape:
                raise ValueError("shape mismatch")
            if not left.any() or not right.any():
                raise ValueError("empty owner")
            if (left & right).any():
                raise ValueError("overlapping owners")
            valid.add(sample_id)
        except Exception as exc:
            rejected[sample_id] = str(exc)
    if not valid:
        raise ValueError(f"no valid masks found under {root}")
    return valid, rejected


def select_rows(
    rows: list[dict], strata: dict[str, int], seed: int, valid_samples: set[str] | None = None
) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if valid_samples is not None and row["sample_id"] not in valid_samples:
            continue
        groups[row["transition"]].append(row)

    selected: list[dict] = []
    used_pairs: set[str] = set()
    for transition, requested in strata.items():
        candidates = groups.get(transition, []).copy()
        rng.shuffle(candidates)
        distinct = [row for row in candidates if row["pair_id"] not in used_pairs]
        fallback = [row for row in candidates if row["pair_id"] in used_pairs]
        picked = (distinct + fallback)[:requested]
        if len(picked) != requested:
            raise ValueError(
                f"transition {transition!r}: requested {requested}, available {len(candidates)}"
            )
        selected.extend(picked)
        used_pairs.update(row["pair_id"] for row in picked)
    return selected


def cohort_for(transition: str) -> str:
    if transition == "both_correct":
        return "false_positive_control"
    if transition == "worsened":
        return "structural_or_binding_risk"
    return "timing_oracle"


def build_manifest(args: argparse.Namespace) -> dict:
    source = load_source_tasks(args.source_manifest)
    transitions = load_transitions(args.transitions_csv)
    valid_samples = None
    rejected_masks: dict[str, str] = {}
    if args.mask_npz_root:
        valid_samples, rejected_masks = valid_mask_samples(args.mask_npz_root)
    selected = select_rows(
        transitions, parse_strata(args.strata), args.selection_seed, valid_samples
    )
    steps = parse_steps(args.intervention_steps, args.num_steps)
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    if not targets or any(target not in {"A", "B", "AB"} for target in targets):
        raise ValueError("targets must be a comma-separated subset of A,B,AB")

    replay_tasks: list[ReplayTask] = []
    for transition_row in selected:
        key = (
            transition_row["pair_id"],
            int(transition_row["seed_group"]),
            int(transition_row["replicate"]),
        )
        if key not in source:
            raise KeyError(f"transition row has no source task: {key}")
        row = source[key]
        common = dict(
            sample_id=transition_row["sample_id"],
            pair_id=row["pair_id"],
            pair_index=int(row["pair_index"]),
            seed_group=int(row["seed_group"]),
            replicate=int(row["replicate"]),
            latent_seed=int(row["latent_seed"]),
            global_prompt=row["global_prompt"],
            entity_A_prompt=row["entity_A_prompt"],
            entity_B_prompt=row["entity_B_prompt"],
            native_task_id=row["native_task_id"],
            source_lsda_task_id=row["task_id"],
            source_transition=transition_row["transition"],
            cohort=cohort_for(transition_row["transition"]),
            split="pilot",
            metadata={
                "native_correct": int(transition_row["native_correct"]),
                "lsda_step0_correct": int(transition_row["lsda_correct"]),
                "selection_seed": args.selection_seed,
            },
        )
        replay_tasks.append(
            ReplayTask(
                replay_id=f'{transition_row["sample_id"]}__wait',
                action=ReplayAction("wait", None, "none"),
                **common,
            )
        )
        for step in steps:
            for target in targets:
                replay_tasks.append(
                    ReplayTask(
                        replay_id=f'{transition_row["sample_id"]}__t{step:02d}__{target}',
                        action=ReplayAction("invoke_lsda", step, target),
                        **common,
                    )
                )

    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.experiment_id,
        "purpose": "counterfactual timing sweep for sequential LSDA tool use",
        "source_manifest": str(args.source_manifest),
        "source_transitions": str(args.transitions_csv),
        "selection": {
            "seed": args.selection_seed,
            "strata": parse_strata(args.strata),
            "selected_samples": len(selected),
            "selected_transition_counts": dict(Counter(row["transition"] for row in selected)),
            "mask_qc_root": str(args.mask_npz_root) if args.mask_npz_root else None,
            "mask_qc_rejected_count": len(rejected_masks),
            "selected_samples_passed_mask_qc": args.mask_npz_root is not None,
        },
        "generation": {
            "model": "stable-diffusion-3.5-large",
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "num_inference_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "intervention_steps": steps,
            "targets": targets,
            "checkpoint_semantics": (
                "WAIT follows the frozen native trajectory. INVOKE starts from the native state "
                "immediately before intervention_step; selected owners follow local experts and "
                "all other owners follow the native trajectory."
            ),
        },
        "task_count": len(replay_tasks),
        "tasks": [task_to_dict(task) for task in replay_tasks],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--transitions-csv", type=Path, required=True)
    parser.add_argument(
        "--mask-npz-root",
        type=Path,
        help="Optional frozen native-SS mask collection used to reject empty/overlapping owners before sampling.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-id", default="2026_9_1_EXP_1")
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--intervention-steps", default="0,4,8,12,16,20,24")
    parser.add_argument("--targets", default="A,B,AB")
    parser.add_argument(
        "--strata",
        default=",".join(f"{name}={count}" for name, count in DEFAULT_STRATA.items()),
        help="Counts by Qwen transition, e.g. restored=5,persistent_failure=3,worsened=2,both_correct=2",
    )
    parser.add_argument("--selection-seed", type=int, default=20260901)
    args = parser.parse_args()
    payload = build_manifest(args)
    write_manifest(payload, args.output)
    print(
        json.dumps(
            {
                "event": "manifest_written",
                "path": str(args.output),
                "selected_samples": payload["selection"]["selected_samples"],
                "task_count": payload["task_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
