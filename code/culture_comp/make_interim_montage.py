"""Interim qualitative montage for EXP_6: SS | Standalone A | Standalone B | LL | LSDA-Long.

Selects samples (from scored intersection) where SS fails and, per mode:
  win : LL fails but LSDA-Long succeeds
  lose: both LL and LSDA-Long fail
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

E = Path(r"D:\Python\MMDIT\experiment")
EXP6 = E / "2026_9_15_EXP_6_KNOW_INJECT"
EXP3 = E / "2026_9_12_EXP_3_KA_ME"
SSDIR = Path(r"D:\Python\MMDIT\data\KA_ME\2026_9_12_EXP_3\jpg")
NEW = Path(r"D:\Python\MMDIT\data\KNOW_INJECT\2026_9_15_EXP_6")


def rows(p):
    d = {}
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            d[r["id"]] = r
    return d


def val(r, rt):
    g = (r or {}).get(rt) or {}
    a, b = g.get("left_is_a"), g.get("right_is_b")
    if a is None and isinstance(b, (int, float)):
        a = 1 - b
    return (a, b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None


def c1(r):
    a, b = val(r, "gpt54"), val(r, "gemini35")
    if not a or not b:
        return None
    return not ((a[0] < .5 or a[1] < .5) and (b[0] < .5 or b[1] < .5))


def font(sz, bold=False):
    for n in (["arialbd.ttf", "arial.ttf"] if bold else ["arial.ttf"]):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    cls = json.loads((EXP3 / "analysis" / "classification.json").read_text(encoding="utf-8"))
    census = rows(EXP3 / "ratings" / "census_scores.jsonl")
    ll = rows(EXP6 / "ratings" / "ll_scores.jsonl")
    lng = rows(EXP6 / "ratings" / "lsdalong_scores.jsonl")
    pairs = json.loads((Path(r"D:\Python\MMDIT\code\mmdit_causal\cultural_pairs_100.json")).read_text(encoding="utf-8"))

    win, lose = [], []
    for rec in cls.values():
        if rec.get("label") not in ("KA", "ME"):
            continue
        i, s = int(rec["pair"]), int(rec["seed"])
        rid = f"p{i:03d}_s{s}"
        ss = c1(census.get(f"cen_{rid}_SS"))
        l1 = c1(ll.get(f"ll_{rid}"))
        l2 = c1(lng.get(f"lsdalong_{rid}"))
        if ss is False and l1 is False and l2 is True:
            win.append((i, s))
        if ss is False and l1 is False and l2 is False:
            lose.append((i, s))
    sel = [("win", x) for x in win[:3]] + [("lose", x) for x in lose[:2]]
    print("selected:", sel)

    tile, bar, head = 300, 44, 30
    cols = 5
    h = head + len(sel) * (bar + tile)
    canvas = Image.new("RGB", (cols * tile, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fh, fc, fs = font(18, True), font(13), font(12)
    for c, name in enumerate(["SS (native)", "Standalone A", "Standalone B", "LL (global long)", "LSDA-Long (ours)"]):
        draw.text((c * tile + 8, 6), name, fill=(31, 78, 121), font=fh)
    y = head
    for mode, (i, s) in sel:
        rec = pairs[i - 1]
        cap = f"[{mode}] p{i:03d}/s{s}  {rec['文化物体A'][:38]}  x  {rec['文化物体B'][:38]}"
        draw.rectangle([0, y, cols * tile, y + bar], fill=(245, 247, 250))
        draw.text((8, y + 12), cap, fill=(51, 64, 85), font=fc)
        y += bar
        paths = [SSDIR / f"pair{i:03d}_seed{s}_SS.jpg", SSDIR / f"pair{i:03d}_seed{s}_A.jpg",
                 SSDIR / f"pair{i:03d}_seed{s}_B.jpg", NEW / "LL" / f"ll_p{i:03d}_s{s}.jpg",
                 NEW / "LSDA_Long" / f"lsdalong_p{i:03d}_s{s}.jpg"]
        for c, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((tile - 6, tile - 6), Image.LANCZOS)
                canvas.paste(im, (c * tile + (tile - im.width) // 2, y + (tile - im.height) // 2))
            draw.rectangle([c * tile, y, c * tile + tile, y + tile], outline=(208, 213, 221))
        y += tile
    out = EXP6 / "figures" / "fig_interim_ll_vs_lsdalong.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
