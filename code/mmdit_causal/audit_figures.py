"""Audit generated figures with gpt-5.6-sol: aesthetics, color, text overlap."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

URL = "https://api.openlux.ai/v1/chat/completions"
MODEL = "gpt-5.6-sol"

PROMPT = (
    "You are a scientific figure QA reviewer for a CVPR paper. Examine this figure carefully. "
    "Check specifically: (1) whether any text, tick label, annotation, or legend overlaps, "
    "collides, or is clipped; (2) whether axis labels and titles are legible; "
    "(3) whether colors are harmonious and distinguishable (colorblind-friendly); "
    "(4) overall aesthetic quality. "
    "Return ONLY JSON:\n"
    "{\n"
    '  "overlap": true/false,\n'
    '  "overlap_details": "where exactly, if any",\n'
    '  "legibility": 0-10,\n'
    '  "color_harmony": 0-10,\n'
    '  "aesthetics": 0-10,\n'
    '  "issues": ["..."],\n'
    '  "suggestions": ["..."]\n'
    "}"
)


def parse_json(text) -> dict:
    if isinstance(text, list):
        text = " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    return json.loads(text[start : end + 1])


def prepare(path: Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(api_key: str, b64: str) -> dict:
    body = {
        "model": MODEL,
        "temperature": 0.0,
        "max_tokens": 700,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last = None
    for delay in (0, 10, 30, 60):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return parse_json(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError,
                json.JSONDecodeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    api_key = os.environ.get("TEST_API_KEY")
    if not api_key:
        raise RuntimeError("TEST_API_KEY missing")

    done = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["figure"]] = r
    with args.out.open("a", encoding="utf-8") as fh:
        for fig in sorted(args.fig_dir.glob("*.png")):
            if fig.name in done:
                continue
            try:
                verdict = call(api_key, prepare(fig))
            except Exception as exc:  # noqa: BLE001
                verdict = {"error": str(exc)}
            row = {"figure": fig.name, **verdict}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({"audited": fig.name, "overlap": verdict.get("overlap"), "aesthetics": verdict.get("aesthetics")}), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
