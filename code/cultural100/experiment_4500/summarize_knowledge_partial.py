from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


path = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500\base_vlm\analysis\seed_level_results.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
knowledge_rows = [row for row in rows if row.get("attribution") == "knowledge_insufficiency"]
groups = defaultdict(lambda: {"seeds": [], "A": 0, "B": 0, "AB": 0})
for row in knowledge_rows:
    key = (row["pair_id"], row["entity_A"], row["entity_B"])
    rec = groups[key]
    rec["seeds"].append(int(row["latent_seed"]))
    a_wrong = (
        row["consensus_A_affected"] == "True"
        and row["qwen_standalone_A_correct"] == "False"
        and row["gemini_standalone_A_correct"] == "False"
    )
    b_wrong = (
        row["consensus_B_affected"] == "True"
        and row["qwen_standalone_B_correct"] == "False"
        and row["gemini_standalone_B_correct"] == "False"
    )
    rec["AB" if a_wrong and b_wrong else "A" if a_wrong else "B" if b_wrong else "AB"] += 1

summary_rows = []
for (pair_id, entity_a, entity_b), rec in sorted(
    groups.items(), key=lambda item: (-len(item[1]["seeds"]), item[0][0])
):
    counts = f'A:{rec["A"]} B:{rec["B"]} AB:{rec["AB"]}'
    summary_rows.append({
        "pair_id": pair_id,
        "n": len(rec["seeds"]),
        "entity_A": entity_a,
        "entity_B": entity_b,
        "seeds": ",".join(map(str, sorted(rec["seeds"]))),
        "A_only_count": rec["A"],
        "B_only_count": rec["B"],
        "AB_count": rec["AB"],
    })

out_dir = path.parent
with (out_dir / "knowledge_insufficiency_seed_list.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=knowledge_rows[0].keys())
    writer.writeheader(); writer.writerows(knowledge_rows)
with (out_dir / "knowledge_insufficiency_by_pair.csv").open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
    writer.writeheader(); writer.writerows(summary_rows)
print(f"knowledge_seed_rows={len(knowledge_rows)} pairs={len(summary_rows)}")
