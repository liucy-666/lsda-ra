"""Select, per pair, the SS seed with the lowest left/right border contact (least cropped),
then emit v15mask jobs JSON using that seed and the given ss_prompt.

Usage:
  python select_ss_seed.py --ss-dir <dir> --pairs-json <cultural_pairs_100.json> \
      --ss-prompt-json <ss_small_pairs.json> --seeds 1011,1012,1013,1014,1015 \
      --out-jobs jobs_v15mask_selected.json [--variant mask]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def border_contact(path: Path) -> tuple[float, float]:
    im = np.asarray(Image.open(path).convert("RGB").resize((128, 128))).astype(np.float32)
    border = np.concatenate([im[:5].reshape(-1, 3), im[-5:].reshape(-1, 3),
                             im[:, :5].reshape(-1, 3), im[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(im - bg[None, None], axis=2) > 60
    return float(fg[:, :3].mean()), float(fg[:, -3:].mean())


def halves(path: Path) -> tuple[float, float]:
    """Foreground fraction of the left / right half: both must be non-empty (two entities)."""
    im = np.asarray(Image.open(path).convert("RGB").resize((128, 128))).astype(np.float32)
    border = np.concatenate([im[:5].reshape(-1, 3), im[-5:].reshape(-1, 3),
                             im[:, :5].reshape(-1, 3), im[:, -5:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(im - bg[None, None], axis=2) > 60
    return float(fg[:, :64].mean()), float(fg[:, 64:].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss-dir", type=Path, required=True)
    ap.add_argument("--pairs-json", type=Path, required=True)
    ap.add_argument("--ss-prompt-json", type=Path, required=True)
    ap.add_argument("--seeds", default="1011,1012,1013,1014,1015")
    ap.add_argument("--out-jobs", type=Path, required=True)
    ap.add_argument("--tag", default="v15mask_sel")
    args = ap.parse_args()

    pairs = json.loads(args.pairs_json.read_text(encoding="utf-8"))
    ss_json = {r["idx"]: r for r in json.loads(args.ss_prompt_json.read_text(encoding="utf-8"))}
    seeds = [int(s) for s in args.seeds.split(",")]

    jobs = []
    print("pair | seed : L/R contact  Lhalf/Rhalf  (selected)")
    for idx in sorted(ss_json):
        v = list(pairs[idx - 1].values())
        best = None
        scores = []
        for s in seeds:
            p = args.ss_dir / f"pair{idx:03d}_seed{s}_SS.png"
            if not p.exists():
                continue
            l, r = border_contact(p)
            lh, rh = halves(p)
            two_obj = (lh > 0.03 and rh > 0.03)
            penalty = 0.0 if two_obj else 1.0
            scores.append((round(l + r + penalty, 3), s, l, r, lh, rh, two_obj))
            if best is None or (l + r + penalty) < best[0]:
                best = (l + r + penalty, s, l, r, lh, rh, two_obj)
        if best is None:
            print(idx, "NO CANDIDATES")
            continue
        for sc, s, l, r, lh, rh, two_obj in sorted(scores):
            flag = "  <== selected" if s == best[1] else ""
            warn = "" if two_obj else "  [ONE-OBJECT]"
            print(f"{idx} | {s} : {l:.2f}/{r:.2f}  {lh:.2f}/{rh:.2f}{warn}{flag}")
        jobs.append({
            "run_id": f"{args.tag}_p{idx:03d}_s{best[1]}", "pair": idx, "seed": best[1], "kind": "AESTH",
            "ss_prompt": ss_json[idx]["ss_prompt"], "a_prompt": v[2], "b_prompt": v[3],
        })
    args.out_jobs.write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out_jobs, "jobs", len(jobs))


if __name__ == "__main__":
    main()
