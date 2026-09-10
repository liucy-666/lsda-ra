from PIL import Image, ImageDraw
from pathlib import Path
d = Path(r"D:\Python\MMDIT\data\LSDA_Timed\2026_9_1_bind")
cell = 384
a = Image.open(d / "natives_sweep" / "native_11.png").convert("RGB").resize((cell, cell))
b = Image.open(d / "seed11" / "bound_imgbind.png").convert("RGB").resize((cell, cell))
sheet = Image.new("RGB", (cell * 2 + 4, cell), "black")
sheet.paste(a, (0, 0)); sheet.paste(b, (cell + 4, 0))
dr = ImageDraw.Draw(sheet)
dr.text((6, 6), "NATIVE (confused)", fill="yellow")
dr.text((cell + 10, 6), "BOUND (rescued)", fill="yellow")
out = d / "seed11_before_after.png"
sheet.save(out)
print("saved", out)
