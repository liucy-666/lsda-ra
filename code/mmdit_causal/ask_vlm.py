"""Ask a vision model about an image (default: gpt-5.6-sol)."""
import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

URL = "https://api.openlux.ai/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-sol"


def parse_json(text):
    if isinstance(text, list):
        text = " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    s, e = text.find("{"), text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            pass
    return {"raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    key = os.environ.get("TEST_API_KEY")
    if not key:
        raise RuntimeError("TEST_API_KEY missing")
    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    body = {
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": 900,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": args.question},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    last = None
    for delay in (0, 10, 30):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            print(payload["choices"][0]["message"]["content"])
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(last)


if __name__ == "__main__":
    main()
