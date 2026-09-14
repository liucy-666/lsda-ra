"""Auto-rank and assemble the qualitative 4-panel demo gallery.

Panels per row: Native SS | Standalone (A | B) | LSDA (ours) | Binding.

Selection is deterministic: for each failure class (KA, ME) rank samples by the VLM
improvement ``min(LSDA sides) - min(SS sides)``, keep those where SS fails and LSDA
succeeds under the project c1 dual-rater criterion, and materialize the top-N per class.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from arms import load_classification, resolve  # noqa: E402

KEYS = ("left_is_a", "right_is_b")


def load_jsonl(path: Path) -> dict:
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def sides(row):
    if not row:
        return None
    g = row.get("gpt54") or {}
    m = row.get("gemini35") or {}
    if not isinstance(g, dict) or not isinstance(m, dict):
        return None
    if not all(k in g for k in KEYS) or not all(k in m for k in KEYS):
        return None
    return [float(g[KEYS[0]]), float(g[KEYS[1]]), float(m[KEYS[0]]), float(m[KEYS[1]])]


def c1_correct(row):
    v = sides(row)
    if v is None:
        return None
    lg, rg, lm, rm = v
    return not ((lg < 0.5 or rg < 0.5) and (lm < 0.5 or rm < 0.5))


def font(size, bold=False):
    names = (["arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "DejaVuSans.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def fit(pil, box):
    bw, bh = box
    w, h = pil.size
    s = min(bw / w, bh / h)
    return pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def paste_center(canvas, pil, box):
    x, y, w, h = box
    im = fit(pil, (w, h))
    canvas.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))


def rank_samples(classification, census_scores, repair_scores, binding_scores,
                 census_dir, lsda_dir, binding_dir, per_class):
    cls = load_classification(classification)
    census = load_jsonl(census_scores)
    rep = {}
    for p in repair_scores:
        rep.update(load_jsonl(p))
    bind = load_jsonl(binding_scores)

    ranked = {"KA": [], "ME": []}
    for rec in cls:
        label = rec["label"]
        if label not in ranked:
            continue
        pair, seed = int(rec["pair"]), int(rec["seed"])
        rid = f"p{pair:03d}_s{seed}"
        arms = resolve(rec, census_dir, lsda_dir, binding_dir)
        ss_row = census.get(f"cen_{rid}_SS")
        lsda_row = rep.get(arms["LSDA"].stem) or census.get(arms["LSDA"].stem)
        bind_row = bind.get(f"bind_{rid}")
        if not all(p.exists() for p in arms.values()):
            continue
        sv, lv = sides(ss_row), sides(lsda_row)
        if sv is None or lv is None:
            continue
        if c1_correct(ss_row) is not False or c1_correct(lsda_row) is not True:
            continue
        effect = min(lv) - min(sv)
        ranked[label].append({
            "label": label, "pair": pair, "seed": seed, "rid": rid,
            "effect": effect, "ss": sv, "lsda": lv, "bind": sides(bind_row),
            "ss_correct": c1_correct(ss_row), "lsda_correct": c1_correct(lsda_row),
            "bind_correct": c1_correct(bind_row), "arms": arms,
        })
    selected = []
    for label in ("KA", "ME"):
        ranked[label].sort(key=lambda r: r["effect"], reverse=True)
        selected.extend(ranked[label][:per_class])
    selected.sort(key=lambda r: (r["label"] != "ME", -r["effect"]))
    return selected


def build(selected, names, out, tile=384):
    pad = 18
    col_label_h = 40
    row_label_h = 34
    status_h = 30
    n = len(selected)
    width = pad * 2 + tile * 4
    height = pad * 2 + col_label_h + n * (row_label_h + tile + status_h)

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_col = font(21, bold=True)
    f_row = font(17, bold=True)
    f_small = font(15)
    f_tiny = font(13)
    f_stat = font(16, bold=True)

    headers = ["Native SS", "Standalone (A | B)", "LSDA (ours)", "Binding"]
    OK, FAIL = "#009E73", "#D55E00"

    for c, htext in enumerate(headers):
        x = pad + c * tile
        draw.rectangle([x, pad, x + tile, pad + col_label_h], fill="#F2F4F7", outline="#D0D5DD")
        tw = draw.textlength(htext, font=f_col)
        draw.text((x + (tile - tw) / 2, pad + (col_label_h - 24) / 2), htext, fill="#1F4E79", font=f_col)

    y = pad + col_label_h
    for row in selected:
        pair, seed, label = row["pair"], row["seed"], row["label"]
        name = names.get(pair, {})
        a = name.get("A", "?")
        b = name.get("B", "?")
        caption = f"{label}  p{pair:03d}/s{seed}   |   {a}   x   {b}"
        draw.rectangle([pad, y, pad + tile * 4, y + row_label_h], fill="#FAFAFA", outline="#E4E7EC")
        draw.text((pad + 10, y + (row_label_h - 20) / 2), caption, fill="#344054", font=f_row)
        y += row_label_h

        cells = [
            ("SS", row["arms"]["SS"], row["ss_correct"]),
            ("Standalone", None, None),
            ("LSDA", row["arms"]["LSDA"], row["lsda_correct"]),
            ("Binding", row["arms"]["Binding"], row["bind_correct"]),
        ]
        for c, (kind, path, ok) in enumerate(cells):
            x = pad + c * tile
            if kind == "Standalone":
                a_img = Image.open(row["arms"]["SS"].with_name(row["arms"]["SS"].name.replace("_SS", "_A"))).convert("RGB")
                b_img = Image.open(row["arms"]["SS"].with_name(row["arms"]["SS"].name.replace("_SS", "_B"))).convert("RGB")
                half = (tile - 6) // 2
                paste_center(canvas, a_img, (x + 2, y, half, tile))
                paste_center(canvas, b_img, (x + 4 + half, y, half, tile))
                draw.text((x + 8, y + 6), "A", fill="#475467", font=f_small)
                draw.text((x + 8 + half + 4, y + 6), "B", fill="#475467", font=f_small)
                draw.line([(x + 3 + half, y + 6), (x + 3 + half, y + tile - 6)], fill="#EAECF0", width=1)
            else:
                paste_center(canvas, Image.open(path).convert("RGB"), (x, y, tile, tile))
            draw.rectangle([x, y, x + tile, y + tile], outline="#D0D5DD")
        y += tile

        for c, (kind, path, ok) in enumerate(cells):
            x = pad + c * tile
            if kind in ("SS", "LSDA", "Binding"):
                if kind == "LSDA":
                    tag, chip = ("CORRECT", OK) if ok is True else ("FAIL", FAIL)
                else:
                    tag, chip = ("FAIL", FAIL) if ok is False else ("CORRECT", OK)
                tw = draw.textlength(tag, font=f_stat)
                draw.rounded_rectangle([x + 6, y + 3, x + 6 + tw + 18, y + 27],
                                       radius=6, fill=chip)
                draw.text((x + 15, y + 6), tag, fill="#FFFFFF", font=f_stat)
            draw.text((x + tile - 92, y + 9), kind, fill="#98A2B3", font=f_tiny)
        y += status_h

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(str(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification", type=Path, required=True)
    ap.add_argument("--census-scores", type=Path, required=True)
    ap.add_argument("--repair-scores", type=Path, nargs="+", required=True)
    ap.add_argument("--binding-scores", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--lsda-dir", type=Path, required=True)
    ap.add_argument("--binding-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--tile", type=int, default=384)
    args = ap.parse_args()

    sys.path.insert(0, r"D:\Python\MMDIT\code\mmdit_causal")
    from pairs100 import PAIRS  # noqa: E402

    selected = rank_samples(
        args.classification, args.census_scores, args.repair_scores, args.binding_scores,
        args.census_dir, args.lsda_dir, args.binding_dir, args.per_class,
    )
    names = {p: {"A": PAIRS[p]["A"], "B": PAIRS[p]["B"]} for p in PAIRS}
    for r in selected:
        print(f"  {r['label']} p{r['pair']:03d}/s{r['seed']} effect={r['effect']:+.2f} "
              f"ss_min={min(r['ss']):.2f} lsda_min={min(r['lsda']):.2f}")
    build(selected, names, args.out, tile=args.tile)


if __name__ == "__main__":
    main()
