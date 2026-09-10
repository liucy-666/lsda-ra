"""Extract CLIP URL + download CSD checkpoint via drive form fields."""
from __future__ import annotations

import re
import urllib.parse
import urllib.request

# 1) find how ViT-L/14 is keyed in clip.py
src = open(r"D:\Python\MMDIT\code\metrics\vendor\CLIP-main\clip\clip.py", encoding="utf-8").read()
for line in src.splitlines():
    if "ViT-L/14" in line:
        print("CLIP_LINE:", line.strip()[:200])

# 2) drive form fields
u = "https://drive.google.com/uc?export=download&id=1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46"
req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
hidden = re.findall(r'<input[^>]*type="hidden"[^>]*>', html)
fields = {}
for h in hidden:
    name = re.search(r'name="([^"]+)"', h)
    val = re.search(r'value="([^"]*)"', h)
    if name:
        fields[name.group(1)] = val.group(1) if val else ""
print("FORM_FIELDS:", fields)
# build download URL like gdown
if "uuid" in fields:
    dl = (
        "https://drive.usercontent.google.com/download?"
        + urllib.parse.urlencode({"id": fields.get("id", "1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46"), "export": "download", "confirm": "t", "uuid": fields["uuid"]})
    )
    print("DL_URL:", dl)
    out = r"D:\Python\MMDIT\models\csd_checkpoint.pt"
    req2 = urllib.request.Request(dl, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req2, timeout=300) as r:
        data = r.read()
    print("downloaded bytes:", len(data))
    with open(out, "wb") as f:
        f.write(data)
    print("saved to", out)
else:
    print("no uuid field -> cannot build confirm URL")
