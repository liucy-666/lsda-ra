"""Phase 1: reconstruct the dual-judge strict failure list for the deep-dive.

Join: binary_vqa_v2 ratings (QWEN/GEMINI) + qwen_image_level.csv (eval_id -> sample metadata).
Strict leakage (per the frozen project rule): a side counts as leaked only if BOTH raters
flag it.  leaked = dual(left_choice == B) OR dual(right_choice == A).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

B = Path(r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1\cultural100_records\experiment_4500\binary_vqa_v2")
OUT = Path(r"D:\Python\MMDIT\experiment\2026_9_10_EXP_1_CAUSAL\deepdive")
TARGET_PAIRS = ["pair_003", "pair_013", "pair_015", "pair_046", "pair_084"]


def load_ratings(rater: str) -> dict:
    rows = {}
    for p in sorted((B / "ratings" / rater).glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["eval_id"]] = r
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(B / "source_data" / "qwen_image_level.csv", encoding="utf-8-sig") as fh:
        meta = {r["eval_id"]: r for r in csv.DictReader(fh)}
    q = load_ratings("QWEN")
    g = load_ratings("GEMINI")

    # per (pair, latent_seed, condition): dual-judge flags
    recs = defaultdict(dict)
    for eval_id, m in meta.items():
        rq, rg = q.get(eval_id), g.get(eval_id)
        if not rq or not rg:
            continue
        left_fail = rq["left_choice"] == "B" and rg["left_choice"] == "B"
        right_fail = rq["right_choice"] == "A" and rg["right_choice"] == "A"
        recs[(m["pair_id"], int(m["latent_seed"]), m["condition"])] = {
            "eval_id": eval_id,
            "sample_id": m["sample_id"],
            "seed_group": int(m["seed_group"]),
            "replicate": int(m["replicate"]),
            "qwen_left": rq["left_choice"], "qwen_right": rq["right_choice"],
            "gemini_left": rg["left_choice"], "gemini_right": rg["right_choice"],
            "left_fail_dual": left_fail, "right_fail_dual": right_fail,
            "leaked": left_fail or right_fail,
        }

    # ---- verify against the frozen 231/900
    native = {k: v for k, v in recs.items() if k[2] == "native_SS"}
    n_fail = sum(1 for v in native.values() if v["leaked"])
    print(f"[verify] native_SS samples={len(native)} dual-strict leaked={n_fail} (frozen reference: 231/900)")

    # ---- per-pair native failure counts
    per_pair = defaultdict(lambda: {"n": 0, "fail": 0, "seeds_fail": [], "seeds_ok": []})
    for (pair, seed, cond), v in sorted(native.items()):
        e = per_pair[pair]
        e["n"] += 1
        if v["leaked"]:
            e["fail"] += 1
            e["seeds_fail"].append(seed)
        else:
            e["seeds_ok"].append(seed)

    ranking = sorted(per_pair.items(), key=lambda kv: -kv[1]["fail"])
    print("\n[ranking] top 15 pairs by dual-strict native failure:")
    for pair, e in ranking[:15]:
        print(f"  {pair}: {e['fail']}/{e['n']} fail | failed seeds={sorted(e['seeds_fail'])}")

    # ---- deep-dive manifest for the 5 target pairs
    manifest = {"target_pairs": TARGET_PAIRS, "rule": "leaked = dual(left==B) OR dual(right==A)", "pairs": {}}
    for pair in TARGET_PAIRS:
        e = per_pair[pair]
        manifest["pairs"][pair] = {
            "n_native": e["n"],
            "n_fail": e["fail"],
            "failed_seeds": sorted(e["seeds_fail"]),
            "normal_seeds": sorted(e["seeds_ok"]),
            "detail": {str(s): native[(pair, s, "native_SS")] for s in sorted(e["seeds_fail"])},
        }
        print(f"\n[{pair}] {e['fail']}/{e['n']} fail")
        print(f"   failed: {sorted(e['seeds_fail'])}")
        print(f"   normal: {sorted(e['seeds_ok'])}")

    (OUT / "dual_strict_native.json").write_text(
        json.dumps({f"{p}|{s}": v for (p, s, c), v in sorted(native.items())}, indent=1), encoding="utf-8"
    )
    (OUT / "deepdive_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\n[saved] {OUT / 'dual_strict_native.json'}")
    print(f"[saved] {OUT / 'deepdive_manifest.json'}")


if __name__ == "__main__":
    main()
