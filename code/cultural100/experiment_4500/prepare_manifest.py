from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment")
SOURCE = ROOT / "cultural_pairs_100.json"
OUT = ROOT / "experiment_4500" / "base_manifest.json"
SEED_GROUPS = (101, 202, 303)
REPLICATES = (1, 2, 3)


def main() -> None:
    pairs = json.loads(SOURCE.read_text(encoding="utf-8"))
    tasks = []
    for pair_index, pair in enumerate(pairs, start=1):
        pair_id = f"pair_{pair_index:03d}"
        prompts = {
            "standalone_A": pair["文化物体A"],
            "standalone_B": pair["文化物体B"],
            "native_SS": pair["组合Prompt SS"],
        }
        for group_index, base_seed in enumerate(SEED_GROUPS, start=1):
            for replicate in REPLICATES:
                latent_seed = base_seed * 10 + replicate
                for image_type, prompt in prompts.items():
                    task_id = f"{pair_id}_{image_type}_g{group_index}_r{replicate}_s{latent_seed}"
                    tasks.append({
                        "task_id": task_id,
                        "pair_id": pair_id,
                        "pair_index": pair_index,
                        "image_type": image_type,
                        "seed_group": group_index,
                        "replicate": replicate,
                        "latent_seed": latent_seed,
                        "prompt": prompt,
                        "entity_A": pair["文化物体A"],
                        "entity_B": pair["文化物体B"],
                        "prompt_SS": pair["组合Prompt SS"],
                    })
    assert len(tasks) == 2700
    assert len({row["task_id"] for row in tasks}) == 2700
    payload = {
        "experiment": "Cultural100 Native Baseline v1",
        "source": str(SOURCE),
        "pair_count": len(pairs),
        "seed_groups": list(SEED_GROUPS),
        "replicates_per_group": len(REPLICATES),
        "latent_seed_rule": "base_seed * 10 + replicate",
        "task_count": len(tasks),
        "model": "stable-diffusion-3.5-large",
        "steps": 28,
        "guidance_scale": 4.5,
        "resolution": [1024, 1024],
        "tasks": tasks,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    print(json.dumps({"output": str(OUT), "tasks": len(tasks), "sha256": digest}))


if __name__ == "__main__":
    main()
