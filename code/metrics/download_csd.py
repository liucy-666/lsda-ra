"""Download the CSD checkpoint from Google Drive (handles the virus-scan confirm page)."""
from __future__ import annotations

import re
import sys
import urllib.parse
import urllib.request

FILE_ID = "1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46"
OUT = r"D:\Python\MMDIT\models\csd_checkpoint.pt"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main() -> None:
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    page = get(url).decode("utf-8", "replace")
    # virus-scan warning -> find confirm token
    m = re.search(r'confirm=([0-9A-Za-z_-]+)', page)
    if m:
        confirm = m.group(1)
        dl = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm={confirm}"
        print(f"confirm token found: {confirm}, downloading...", flush=True)
        data = get(dl)
    else:
        # maybe direct download
        print("no confirm token, trying direct...", flush=True)
        data = get(url)
    with open(OUT, "wb") as f:
        f.write(data)
    print(f"downloaded {len(data)/2**20:.1f} MB -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
