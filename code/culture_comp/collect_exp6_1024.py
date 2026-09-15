"""Collect 1024x1024 JPGs (NO downscale) for EXP_6 figures/eval.

- data/KNOW_INJECT/.../LL/*.png            -> collected1024/LL/<id>.jpg
- data/KNOW_INJECT/.../LSDA_Long/runs/*    -> collected1024/LSDA_Long/<id>.jpg
- exp3 census A/B/SS (selected pairs)      -> collected1024/refs/<name>.jpg
- exp3 repair/repair_meattrs lsda.png      -> collected1024/lsda_attrs/<id>.jpg
"""
from pathlib import Path
from PIL import Image

ROOT = Path("/science/wx/pry/MMDIT")
BASE = ROOT / "data/KNOW_INJECT/2026_9_15_EXP_6"
EXP3 = ROOT / "experiments/2026_9_12_EXP_3_KA_ME"
OUT = BASE / "collected1024"
PAIRS = [2, 3, 4, 12, 13]
SEEDS = [1011, 1012, 1013]


def save(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    im.save(dst, quality=92)


def main() -> None:
    n = 0
    for p in sorted((BASE / "LL").glob("*.png")):
        save(p, OUT / "LL" / f"{p.stem}.jpg")
        n += 1
    for d in sorted((BASE / "LSDA_Long" / "runs").iterdir()):
        src = d / "lsda.png"
        if src.exists():
            save(src, OUT / "LSDA_Long" / f"{d.name}.jpg")
            n += 1
    for i in PAIRS:
        for s in SEEDS:
            for kind in ("A", "B", "SS"):
                src = EXP3 / "census" / "images" / f"pair{i:03d}_seed{s}_{kind}.png"
                if src.exists():
                    save(src, OUT / "refs" / f"pair{i:03d}_seed{s}_{kind}.jpg")
                    n += 1
    for i, s in ((4, 1011), (12, 1013)):
        src = EXP3 / "repair" / "runs" / f"lsda_ka_p{i:03d}_s{s}" / "lsda.png"
        if src.exists():
            save(src, OUT / "lsda_attrs" / f"lsda_ka_p{i:03d}_s{s}.jpg")
            n += 1
    for i, s in ((13, 1011), (2, 1012), (3, 1012)):
        src = EXP3 / "repair_meattrs" / "runs" / f"lsda_meattrs_p{i:03d}_s{s}" / "lsda.png"
        if src.exists():
            save(src, OUT / "lsda_attrs" / f"lsda_meattrs_p{i:03d}_s{s}.jpg")
            n += 1
    print("collected1024", n)


if __name__ == "__main__":
    main()
