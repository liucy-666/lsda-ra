"""Phase 1a: explore the join between Qwen/Gemini ratings and sample metadata."""
import json
from pathlib import Path

B = Path(r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1\cultural100_records\experiment_4500\binary_vqa_v2")

oq = json.loads((B / "order_QWEN.json").read_text(encoding="utf-8"))
og = json.loads((B / "order_GEMINI.json").read_text(encoding="utf-8"))
print("order_QWEN:", type(oq).__name__, len(oq), "first:", oq[:2])
print("order_GEMINI:", type(og).__name__, len(og), "first:", og[:2])
print("orders identical:", oq == og)


def load_ratings(rater):
    rows = {}
    for p in sorted((B / "ratings" / rater).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["eval_id"]] = r
    return rows


q = load_ratings("QWEN")
g = load_ratings("GEMINI")
print("Qwen ratings:", len(q), "Gemini ratings:", len(g))
print("eval_id overlap:", len(set(q) & set(g)))
print("Qwen-only:", len(set(q) - set(g)), "Gemini-only:", len(set(g) - set(q)))

# qwen_image_level coverage
import csv
with open(B / "source_data" / "qwen_image_level.csv", encoding="utf-8-sig") as fh:
    img = list(csv.DictReader(fh))
print("qwen_image_level rows:", len(img))
img_ids = {r["eval_id"] for r in img}
print("img-level ids in Qwen ratings:", len(img_ids & set(q)))
print("sample of img-level eval_ids:", sorted(img_ids)[:4])
print("sample of Qwen rating ids:", sorted(q)[:4])
print("sample of Gemini rating ids:", sorted(g)[:4])
