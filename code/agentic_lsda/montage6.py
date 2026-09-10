from PIL import Image, ImageDraw
from pathlib import Path
d = Path(r"D:\Python\MMDIT\data\LSDA_Timed\2026_9_1_bind\concept_test")
order = [("mojolica_alone.png", "Mojolica alone"), ("maiolica_alone.png", "Maiolica alone"),
         ("bluewhite_alone.png", "Chinese blue-white alone")]
cell = 320
sheet = Image.new("RGB", (cell * 3 + 8, cell), "black")
dr = ImageDraw.Draw(sheet)
for i, (f, label) in enumerate(order):
    img = Image.open(d / f).convert("RGB").resize((cell, cell))
    x = i * (cell + 4)
    sheet.paste(img, (x, 0))
    dr.text((x + 6, 6), label, fill="yellow")
sheet.save(d / "concept_sheet.png")
print("saved")
