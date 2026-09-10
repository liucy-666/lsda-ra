"""C1/C2 single-image scorer for pilot counterfactual branches.

For each replay branch image, asks Qwen + Gemini (openlux OpenAI-compatible
endpoint) two questions:
  C1 binding  : left object vs entity_A style, right object vs entity_B style
                (0-1 each; 1 = perfectly bound to its intended culture)
  C2 structure: exactly two entities, no ghosting/seams/contour damage (0-1)
Outputs scores.jsonl: replay_id -> {binding_A, binding_B, structure,
artifact_flags, qwen, gemini, agreement}.

Usage:
  python score_pilot_c1c2.py --manifest <pilot.json> --images-root <dir> \
      [--limit N] [--out scores.jsonl]
Reads image per replay_id from --images-root/{replay_id}.png
(resume: skips replay_ids already in --out)
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
from pathlib import Path

URL = "https://api.openlux.ai/v1/chat/completions"
MODELS = {"QWEN": "qwen3-vl-235b-a22b-instruct", "GEMINI": "gemini-3.5-flash-lite"}


def build_prompt(task: dict) -> str:
    a = task["entity_A_prompt"]
    b = task["entity_B_prompt"]
    scene = task["global_prompt"]
    return (
        "You are evaluating one AI-generated image. The intended scene is:\n"
        f'"{scene}"\n'
        f"The LEFT object should be: {a}\n"
        f"The RIGHT object should be: {b}\n\n"
        "Answer about the ACTUAL image, not the intent. Return ONLY JSON:\n"
        "{\n"
        '  "left_is_a": 0.0,   // 0-1: how much the LEFT object matches A culture/style (1 = perfect)\n'
        '  "right_is_b": 0.0,  // 0-1: how much the RIGHT object matches B culture/style (1 = perfect)\n'
        '  "structure": 0.0,   // 0-1: exactly two separate entities, no ghosting/duplicates/seams/contour damage\n'
        '  "entity_count": 2,  // number of distinct objects visible\n'
        '  "artifacts": ["ghost", "seam", "duplicate", "none"],\n'
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


def find_image(root: Path, rid: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        cand = root / f"{rid}{ext}"
        if cand.exists():
            return cand
    return None


def mime_of(path: Path) -> str:
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
        path.suffix.lower(), "image/png"
    )


def prepare_image(path: Path, max_side: int) -> tuple[str, str]:
    """Return (base64 str, mime). Resize + JPEG when max_side > 0."""
    if max_side <= 0:
        return base64.b64encode(path.read_bytes()).decode(), mime_of(path)
    from io import BytesIO
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
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
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
    for delay in (0, 5, 20, 60):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            return parse_json(payload["choices"][0]["message"]["content"])
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("scores_pilot.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--resize", type=int, default=768)
    ap.add_argument("--raters", default="QWEN,GEMINI")
    args = ap.parse_args()

    api_key = os.environ.get("TEST_API_KEY")
    if not api_key:
        raise RuntimeError("TEST_API_KEY missing")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = payload["tasks"][args.start:]
    if args.limit:
        tasks = tasks[: args.limit]

    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["replay_id"])

    raters = [r for r in args.raters.split(",") if r]
    with args.out.open("a", encoding="utf-8") as fh:
        for task in tasks:
            rid = task["replay_id"]
            if rid in done:
                continue
            img = find_image(args.images_root, rid)
            if img is None:
                continue
            img_b64, mime = prepare_image(img, args.resize)
            prompt = build_prompt(task)
            row = {"replay_id": rid, "sample_id": task["sample_id"],
                   "target": task["action"]["target"],
                   "step": task["action"]["intervention_step"]}
            ok = True
            for rater in raters:
                try:
                    r = call(api_key, rater, img_b64, mime, prompt)
                    row[rater] = {
                        "left_is_a": float(r.get("left_is_a", 0.0)),
                        "right_is_b": float(r.get("right_is_b", 0.0)),
                        "structure": float(r.get("structure", 0.0)),
                        "entity_count": int(r.get("entity_count", -1)),
                        "artifacts": r.get("artifacts", []),
                    }
                except Exception as exc:
                    row[rater] = {"error": f"{type(exc).__name__}: {exc}"}
                    ok = False
            if ok:
                q, g = row.get("QWEN"), row.get("GEMINI")
                row["binding_A"] = (q["left_is_a"] + g["left_is_a"]) / 2
                row["binding_B"] = (q["right_is_b"] + g["right_is_b"]) / 2
                row["structure"] = (q["structure"] + g["structure"]) / 2
                row["binding_agree"] = abs(q["right_is_b"] - g["right_is_b"]) < 0.3
                row["structure_agree"] = abs(q["structure"] - g["structure"]) < 0.3
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({"event": "scored", "replay_id": rid}), flush=True)


if __name__ == "__main__":
    main()
