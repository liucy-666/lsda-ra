"""Qualitative montage incl. LSDA-attrs: SS | Standalone A | Standalone B | LL | LSDA-attrs | LSDA-Long."""
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
    rep = rows(EXP3 / "ratings" / "repair_scores.jsonl")
    meattrs = rows(EXP3 / "ratings" / "meattrs_scores.jsonl")
    ll = rows(EXP6 / "ratings" / "ll_scores.jsonl")
    lng = rows(EXP6 / "ratings" / "lsdalong_scores.jsonl")
    pairs = json.loads(Path(r"D:\Python\MMDIT\code\mmdit_causal\cultural_pairs_100.json").read_text(encoding="utf-8"))

    win, differ = [], []
    for rec in cls.values():
        if rec.get("label") not in ("KA", "ME"):
            continue
        i, s, lab = int(rec["pair"]), int(rec["seed"]), rec["label"]
        rid = f"p{i:03d}_s{s}"
        ss = c1(census.get(f"cen_{rid}_SS"))
        l1 = c1(ll.get(f"ll_{rid}"))
        l2 = c1(lng.get(f"lsdalong_{rid}"))
        a = c1(meattrs.get(f"lsda_meattrs_{rid}" ) if lab == "ME" else rep.get(f"lsda_ka_{rid}"))
        if None in (ss, l1, l2, a):
            continue
        if ss is False and l1 is False and l2 is True and a is True:
            win.append((i, s, lab, ss, l1, a, l2))
        if ss is False and a != l2:
            differ.append((i, s, lab, ss, l1, a, l2))
    sel = [("win", x) for x in win[:3]] + [("attrs-vs-long", x) for x in differ[:2]]
    print("selected:", [(m, x[0], x[1], x[2]) for m, x in sel])

    attrs_path = lambda i, s, lab, rid: SSDIR / (f"lsda_meattrs_{rid}.jpg" if lab == "ME" else f"lsda_ka_{rid}.jpg")
    tile, bar, head = 360, 62, 34
    cols = 6
    canvas = Image.new("RGB", (cols * tile, head + len(sel) * (bar + tile)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    fh, fc, fs = font(17, True), font(13), font(13, True)
    heads = ["SS (native)", "Standalone A", "Standalone B", "LL (global long)", "LSDA-attrs", "LSDA-Long"]
    for c, name in enumerate(heads):
        draw.text((c * tile + 8, 8), name, fill=(31, 78, 121), font=fh)
    y = head
    for mode, (i, s, lab, ss, l1, a, l2) in sel:
        rec = pairs[i - 1]
        a_name = rec["文化物体A"][:52]
        b_name = rec["文化物体B"][:52]
        title = f"{mode} · p{i:03d}/s{s} · {lab} · {a_name}  ×  {b_name}"
        draw.rectangle([0, y, cols * tile, y + bar], fill=(245, 247, 250))
        draw.text((10, y + 9), title, fill=(40, 52, 70), font=fc)
        x = 10
        for label, ok in (("SS", ss), ("LL", l1), ("LSDA-attrs", a), ("LSDA-Long", l2)):
            txt = f"{label}={'pass' if ok else 'fail'}"
            col = (31, 122, 90) if ok else (176, 65, 62)
            draw.text((x, y + 33), txt, fill=col, font=fs)
            x += int(draw.textlength(txt, font=fs)) + 26
        y += bar
        rid = f"p{i:03d}_s{s}"
        paths = [SSDIR / f"pair{i:03d}_seed{s}_SS.jpg", SSDIR / f"pair{i:03d}_seed{s}_A.jpg",
                 SSDIR / f"pair{i:03d}_seed{s}_B.jpg", NEW / "LL" / f"ll_{rid}.jpg",
                 attrs_path(i, s, lab, rid), NEW / "LSDA_Long" / f"lsdalong_{rid}.jpg"]
        for c, p in enumerate(paths):
            if p.exists():
                im = Image.open(p).convert("RGB")
                im.thumbnail((tile - 6, tile - 6), Image.LANCZOS)
                canvas.paste(im, (c * tile + (tile - im.width) // 2, y + (tile - im.height) // 2))
            draw.rectangle([c * tile, y, c * tile + tile, y + tile], outline=(208, 213, 221))
        y += tile
    out = EXP6 / "figures" / "fig_attrs_vs_long_montage.png"
    canvas.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
