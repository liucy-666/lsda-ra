"""Collect EXP_6 generated images into flat 768px JPG dirs for transfer/scoring."""
from pathlib import Path
from PIL import Image

BASE = Path("/science/wx/pry/MMDIT/data/KNOW_INJECT/2026_9_15_EXP_6")
OUT = BASE / "collected"


def save(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    w, h = im.size
    s = 768 / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    im.save(dst, quality=88)


def main() -> None:
    n = {"LL": 0, "LSDA_Long": 0}
    for p in sorted((BASE / "LL").glob("*.png")):
        save(p, OUT / "LL" / f"{p.stem}.jpg")
        n["LL"] += 1
    for d in sorted((BASE / "LSDA_Long" / "runs").iterdir()):
        src = d / "lsda.png"
        if src.exists():
            save(src, OUT / "LSDA_Long" / f"{d.name}.jpg")
            n["LSDA_Long"] += 1
    print(n)


if __name__ == "__main__":
    main()
