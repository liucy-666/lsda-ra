from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
BASE = ROOT / "base_manifest.json"
OUT = ROOT / "lsda_manifest.json"


def main() -> None:
    payload = json.loads(BASE.read_text(encoding="utf-8"))
    index = {
        (row["pair_id"], row["seed_group"], row["replicate"], row["image_type"]): row
        for row in payload["tasks"]
    }
    tasks = []
    for row in payload["tasks"]:
        if row["image_type"] != "native_SS":
            continue
        key = (row["pair_id"], row["seed_group"], row["replicate"])
        a = index[(*key, "standalone_A")]
        b = index[(*key, "standalone_B")]
        tasks.append({
            "task_id": row["task_id"].replace("native_SS", "lsda_ra"),
            "pair_id": row["pair_id"],
            "pair_index": row["pair_index"],
            "seed_group": row["seed_group"],
            "replicate": row["replicate"],
            "latent_seed": row["latent_seed"],
            "global_prompt": row["prompt_SS"],
            "entity_A_prompt": row["entity_A"],
            "entity_B_prompt": row["entity_B"],
            "native_task_id": row["task_id"],
            "standalone_A_task_id": a["task_id"],
            "standalone_B_task_id": b["task_id"],
        })
    assert len(tasks) == 900
    assert len({row["task_id"] for row in tasks}) == 900
    out = {
        "experiment": "Cultural100 LSDA-RA standalone-short v1",
        "task_count": 900,
        "external_visual_information": "standalone Short A/B images only",
        "sl_ll_information_used": False,
        "handoff_step": 7,
        "early_local_weights": [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00],
        "tasks": tasks,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "tasks": len(tasks)}))


if __name__ == "__main__":
    main()
