from PIL import Image, ImageDraw
from pathlib import Path
d = Path(r"D:\Python\MMDIT\data\LSDA_Timed\2026_9_1_bind\mojolica")
seeds = [11, 15, 23, 29]
cell = 384
cols = 2
rows = 2
sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
dr = ImageDraw.Draw(sheet)
for i, s in enumerate(seeds):
    img = Image.open(d / f"native_{s}.png").convert("RGB").resize((cell, cell))
    x, y = (i % cols) * cell, (i // cols) * cell
    sheet.paste(img, (x, y))
    dr.rectangle([x, y, x + cell - 1, y + cell - 1], outline="black")
    dr.text((x + 6, y + 6), f"seed {s}", fill="yellow")
sheet.save(d / "contact.png")
print("saved")
