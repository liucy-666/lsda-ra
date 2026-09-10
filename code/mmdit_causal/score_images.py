"""Score generated cultural-binding images with Qwen + Gemini (openlux).

Manifest: JSONL, one record per image:
  {"id": str, "image": abs_path, "mode": "single"|"pair",
   "desc": str (single), "left": str, "right": str (pair)}

Output JSONL: {"id", "qwen": {...}, "gemini": {...}, "agree_left", ...}
"""
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
MODELS = {"QWEN": "qwen3-vl-235b-a22b-instruct", "GEMINI": "gemini-3.5-flash-lite"}

SINGLE_PROMPT = (
    "You are evaluating one AI-generated image. The intended object is:\n"
    '"<<DESC>>"\n\n'
    "Answer about the ACTUAL image, not the intent. Return ONLY JSON:\n"
    "{\n"
    '  "matches": 0.0,   // 0-1: how much the object matches the described culture/style (1 = perfect)\n'
    '  "structure": 0.0, // 0-1: one clean object, no ghosting/duplicates/contour damage\n'
    '  "reason": "one short sentence"\n'
    "}\n"
    "Compare ONLY surface cultural appearance (material, color, texture, motifs, technique); "
    "ignore silhouette/shape/pose."
)

PAIR_PROMPT = (
    "You are evaluating one AI-generated image. The intended scene is:\n"
    '"<<SCENE>>"\n'
    "The LEFT object should be: <<LEFT>>\n"
    "The RIGHT object should be: <<RIGHT>>\n\n"
    "Answer about the ACTUAL image, not the intent. Return ONLY JSON:\n"
    "{\n"
    '  "left_is_a": 0.0,   // 0-1: how much the LEFT object matches A culture/style (1 = perfect)\n'
    '  "right_is_b": 0.0,  // 0-1: how much the RIGHT object matches B culture/style (1 = perfect)\n'
    '  "structure": 0.0,   // 0-1: exactly two separate entities, no ghosting/seams/contour damage\n'
    '  "entity_count": 2,\n'
    '  "reason": "one short sentence"\n'
    "}\n"
    "Compare ONLY surface cultural appearance (material, color, texture, motifs, technique); "
    "ignore silhouette/shape/pose."
)


def parse_json(text) -> dict:
    if isinstance(text, list):
        text = " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in text)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start : end + 1])


def prepare_image(path: Path, max_side: int) -> tuple[str, str]:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def call(api_key: str, rater: str, image_b64: str, mime: str, prompt: str) -> dict:
    body = {
        "model": MODELS[rater],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
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
    for delay in (0, 8, 25, 60, 120):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return parse_json(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError,
                json.JSONDecodeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


def build_prompt(rec: dict) -> str:
    if rec["mode"] == "single":
        return SINGLE_PROMPT.replace("<<DESC>>", rec["desc"])
    return (
        PAIR_PROMPT.replace("<<SCENE>>", rec.get("scene", ""))
        .replace("<<LEFT>>", rec["left"])
        .replace("<<RIGHT>>", rec["right"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--raters", default="QWEN,GEMINI")
    ap.add_argument("--resize", type=int, default=768)
    ap.add_argument("--sleep", type=float, default=2.0)
    args = ap.parse_args()

    api_key = os.environ.get("TEST_API_KEY")
    if not api_key:
        raise RuntimeError("TEST_API_KEY missing")

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])

    raters = [r.strip().upper() for r in args.raters.split(",") if r.strip()]
    records = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    with args.out.open("a", encoding="utf-8") as fh:
        for rec in records:
            if rec["id"] in done:
                continue
            img_path = Path(rec["image"])
            if not img_path.exists():
                print(json.dumps({"skip": rec["id"], "reason": "missing"}), flush=True)
                continue
            b64, mime = prepare_image(img_path, args.resize)
            prompt = build_prompt(rec)
            row = {"id": rec["id"], "image": str(img_path)}
            for rater in raters:
                try:
                    row[rater.lower()] = call(api_key, rater, b64, mime, prompt)
                except Exception as exc:  # noqa: BLE001
                    row[rater.lower()] = {"error": str(exc)}
                time.sleep(args.sleep)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({"scored": rec["id"]}), flush=True)


if __name__ == "__main__":
    main()
