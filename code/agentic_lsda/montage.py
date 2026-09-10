from PIL import Image, ImageDraw
from pathlib import Path

d = Path(r"D:\Python\MMDIT\data\LSDA_Timed\2026_9_1_bind\natives_sweep")
seeds = [1011, 3, 7, 11, 15, 17, 23, 29]
cell = 256
cols = 2
rows = (len(seeds) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
dr = ImageDraw.Draw(sheet)
for i, s in enumerate(seeds):
    img = Image.open(d / f"native_{s}.png").convert("RGB").resize((cell, cell))
    x, y = (i % cols) * cell, (i // cols) * cell
    sheet.paste(img, (x, y))
    dr.rectangle([x, y, x + cell - 1, y + cell - 1], outline="black")
    dr.text((x + 4, y + 4), str(s), fill="yellow")
sheet.save(d / "contact_sheet.png")
print("saved", sheet.size)
