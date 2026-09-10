from PIL import Image, ImageDraw
from pathlib import Path
d = Path(r"D:\Python\MMDIT\data\LSDA_Timed\2026_9_1_bind\seed11_bw")
cell = 384
a = Image.open(d / "native.png").convert("RGB").resize((cell, cell))
b = Image.open(d / "bound.png").convert("RGB").resize((cell, cell))
sheet = Image.new("RGB", (cell * 2 + 4, cell), "black")
sheet.paste(a, (0, 0)); sheet.paste(b, (cell + 4, 0))
dr = ImageDraw.Draw(sheet)
dr.text((6, 6), "NATIVE", fill="yellow"); dr.text((cell + 10, 6), "BOUND", fill="yellow")
sheet.save(d / "before_after.png")
print("saved")
