from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"D:\Python\MMDIT\AAA_Experiment\experiment_4500")
OUT = ROOT / "independent_pilot" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)
ROWS = [
    (
        "sample_0068 | pair_061 | seed 2023",
        "sample_0068",
        ROOT / "independent_pilot/images/pair_061/pair_061_lsda_independent_g2_r3_s2023.png",
    ),
    (
        "sample_0244 | pair_007 | seed 2022",
        "sample_0244",
        ROOT / "independent_pilot/images/pair_007/pair_007_lsda_independent_g2_r2_s2022.png",
    ),
    (
        "sample_0405 | pair_098 | seed 2022",
        "sample_0405",
        ROOT / "independent_pilot/images/pair_098/pair_098_lsda_independent_g2_r2_s2022.png",
    ),
]
LABELS = ("A-alone", "B-alone", "Native SS", "Corrected LSDA")
PAGE_W = 2400
MARGIN = 28
TITLE_H = 54
GAP = 28


def font(size):
    path = Path(r"C:\Windows\Fonts\arial.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def row_image(sample_id, corrected_path):
    base = Image.open(ROOT / "base_vlm/blind" / f"{sample_id}.jpg").convert("RGB")
    cell = base.width // 3
    panels = [
        base.crop((0, 0, cell, base.height)),
        base.crop((cell, 0, 2 * cell, base.height)),
        base.crop((2 * cell, 0, 3 * cell, base.height)),
        Image.open(corrected_path).convert("RGB").resize((cell, base.height), Image.Resampling.LANCZOS),
    ]
    sheet = Image.new("RGB", (4 * cell, base.height), "#FAF6EE")
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(zip(panels, LABELS)):
        sheet.paste(panel, (index * cell, 0))
        draw.rectangle((index * cell, 0, index * cell + 250, 42), fill="#FAF6EE")
        draw.text((index * cell + 12, 7), label, fill="#1F4E79", font=font(24))
    return sheet


def main():
    tile_w = PAGE_W - 2 * MARGIN
    tile_h = round(tile_w / 4)
    page_h = 2 * MARGIN + len(ROWS) * (TITLE_H + tile_h) + (len(ROWS) - 1) * GAP
    page = Image.new("RGB", (PAGE_W, page_h), "#FAF6EE")
    draw = ImageDraw.Draw(page)
    for index, (title, sample_id, corrected_path) in enumerate(ROWS):
        y = MARGIN + index * (TITLE_H + tile_h + GAP)
        draw.text((MARGIN, y + 7), title, fill="#1F4E79", font=font(30))
        sheet = row_image(sample_id, corrected_path).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        page.paste(sheet, (MARGIN, y + TITLE_H))
        sheet.save(OUT / f"{sample_id}_comparison.jpg", quality=95, subsampling=0)
    page.save(OUT / "pilot_comparison_all.jpg", quality=95, subsampling=0)
    print(OUT / "pilot_comparison_all.jpg")


if __name__ == "__main__":
    main()
