"""Worked example for Method 3.3: native SS vs unified-knowledge LSDA, with the injected text.

Controlled single case (pair 081, seed 1012): the region experts read
``short prompt + rule-extracted attribute phrases`` instead of the short prompt alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def wrap(draw, text, fnt, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss", required=True)
    ap.add_argument("--lsda", required=True)
    ap.add_argument("--attrs", type=Path, required=True)
    ap.add_argument("--pair", default="081")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=430)
    args = ap.parse_args()

    tile = args.tile
    pad = 18
    col_label_h = 40
    text_w = 470
    height = pad * 2 + col_label_h + tile
    width = pad * 2 + tile * 2 + text_w

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    f_col = font(21, bold=True)
    f_title = font(17, bold=True)
    f_body = font(15)
    f_tag = font(16, bold=True)

    headers = ["Native SS", "LSDA (ours)"]
    paths = [args.ss, args.lsda]
    for c, (htext, path) in enumerate(zip(headers, paths)):
        x = pad + c * tile
        draw.rectangle([x, pad, x + tile, pad + col_label_h], fill="#F2F4F7", outline="#D0D5DD")
        tw = draw.textlength(htext, font=f_col)
        draw.text((x + (tile - tw) / 2, pad + (col_label_h - 24) / 2), htext, fill="#1F4E79", font=f_col)
        y = pad + col_label_h
        im = fit(Image.open(path).convert("RGB"), (tile, tile))
        canvas.paste(im, (x + (tile - im.width) // 2, y + (tile - im.height) // 2))
        draw.rectangle([x, y, x + tile, y + tile], outline="#D0D5DD")
        tag, chip = ("FAIL", "#D55E00") if c == 0 else ("CORRECT", "#009E73")
        tw2 = draw.textlength(tag, font=f_tag)
        draw.rounded_rectangle([x + 8, y + 6, x + 8 + tw2 + 18, y + 32], radius=6, fill=chip)
        draw.text((x + 17, y + 9), tag, fill="#FFFFFF", font=f_tag)

    attrs = json.loads(args.attrs.read_text(encoding="utf-8"))[args.pair]
    tx = pad + tile * 2 + 18
    blocks = [
        ("A  short", attrs["A"]["short"], "#475467"),
        ("A  + attrs", attrs["A"]["attrs"], "#2F9E6E"),
        ("B  short", attrs["B"]["short"], "#475467"),
        ("B  + attrs", attrs["B"]["attrs"], "#2F9E6E"),
    ]
    wrapped = [(h, wrap(draw, b, f_body, text_w - 24), col) for h, b, col in blocks]
    block_h = 30 + sum(24 + len(lines) * 22 + 12 for _, lines, _ in wrapped)
    ty = pad + col_label_h + max(6, (tile - block_h) / 2)
    draw.text((tx, ty), "Unified knowledge injection", fill="#1F4E79", font=f_title)
    ty += 30
    for head, lines, color in wrapped:
        draw.text((tx, ty), head, fill=color, font=f_title)
        ty += 24
        for line in lines:
            draw.text((tx, ty), line, fill="#344054", font=f_body)
            ty += 22
        ty += 12

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(str(args.out))


if __name__ == "__main__":
    main()
