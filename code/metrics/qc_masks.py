"""Quantitative QC of SAM masks: centroid, area, overlap, empties."""
from __future__ import annotations

from pathlib import Path

import numpy as np

SEG = Path(r"D:\Python\MMDIT\experiment\2026_8_31_EXP_2\segmentation")


def info(a: np.ndarray) -> dict | None:
    ys, xs = np.where(a > 0)
    if len(xs) == 0:
        return None
    return {"cx": float(xs.mean() / a.shape[1]), "area": float(a.mean())}


def main() -> None:
    for cond in ["native_SS", "lsda_clean", "standalone_A", "standalone_B"]:
        d = SEG / f"masks_{cond}"
        files = sorted(d.glob("*_mask.npz"))
        is_standalone = cond.startswith("standalone")
        bad: list[tuple[str, str]] = []
        for f in files:
            m = np.load(f)
            L, R = m["left"], m["right"]
            li, ri = info(L), info(R)
            issue = []
            if li is None or li["area"] < 0.02:
                issue.append(f"L-empty({0 if li is None else round(li['area'],3)})")
            elif not li["cx"] < 0.55:
                issue.append(f"L-cx={li['cx']:.2f}")
            if not is_standalone:
                if ri is None or ri["area"] < 0.02:
                    issue.append(f"R-empty({0 if ri is None else round(ri['area'],3)})")
                elif not ri["cx"] > 0.45:
                    issue.append(f"R-cx={ri['cx']:.2f}")
            overlap = float((L & R).mean())
            if overlap > 0.02:
                issue.append(f"overlap={overlap:.3f}")
            if issue:
                bad.append((f.name.replace("_mask.npz", ""), ";".join(issue)))
        print(f"{cond}: {len(files)} masks, issues: {len(bad)}")
        for name, why in bad[:15]:
            print(f"   {name}: {why}")


if __name__ == "__main__":
    main()
