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


ROOT = Path(
    r"D:\Python\MMDIT\experiment\2026_8_25_EXP_1"
    r"\cultural100_records\experiment_4500\binary_vqa_v2"
)
IMAGE_ROOT = Path(r"D:\Python\MMDIT\data\Blind\2026_8_25_EXP_1")
URL = "https://api.openlux.ai/v1/chat/completions"


def parse_json(text):
    if isinstance(text, list):
        text = " ".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in text
        )
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    result = json.loads(text[start : end + 1])
    if result.get("right_choice") not in ("A", "B"):
        raise ValueError("invalid right_choice")
    if result.get("left_choice") not in ("A", "B"):
        raise ValueError("invalid left_choice")
    return result


def prompt(eval_id: str) -> str:
    return f"""This is a blind pairwise visual attribution task. The image contains three labeled panels: A-alone, B-alone, and Candidate AB.

Compare ONLY surface cultural appearance: material impression, color system, texture, motifs, ornament, decorative technique, and finish. IGNORE physical silhouette, shape, size, pose, relative scale, and object count.

Question 1: Does the RIGHT entity in Candidate AB look more similar in surface style and texture to A-alone or B-alone?
Question 2: Does the LEFT entity in Candidate AB look more similar in surface style and texture to A-alone or B-alone?

You must choose exactly A or B for each question. Return only this JSON object:
{{"eval_id":"{eval_id}","right_choice":"A|B","left_choice":"A|B","right_reason":"one short visual reason","left_reason":"one short visual reason"}}"""


def call(api_key: str, model: str, eval_id: str):
    jpg = (IMAGE_ROOT / f"{eval_id}.jpg").read_bytes()
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt(eval_id)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,"
                            + base64.b64encode(jpg).decode("ascii")
                        },
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last = None
    for delay in (0, 5, 20, 60, 180):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return parse_json(payload["choices"][0]["message"]["content"])
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


def existing_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("eval_id"):
                    ids.add(row["eval_id"])
            except json.JSONDecodeError:
                continue
    return ids


def all_existing_ids(rater: str) -> set[str]:
    ids: set[str] = set()
    for path in sorted((ROOT / "ratings" / rater).glob("ratings_*.jsonl")):
        ids.update(existing_ids(path))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater", choices=("GEMINI", "QWEN"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--nchunks", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    api_key = os.environ.get("TEST_API_KEY")
    if not api_key:
        raise RuntimeError("TEST_API_KEY missing")

    order = json.loads((ROOT / f"order_{args.rater}.json").read_text(encoding="utf-8"))
    ids = order[args.chunk :: args.nchunks]
    if args.limit:
        ids = ids[: args.limit]
    out_dir = ROOT / "ratings" / args.rater
    log_dir = ROOT / "logs" / args.rater
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ratings_{args.rater}_chunk_{args.chunk:02d}.jsonl"
    errors = log_dir / f"errors_{args.rater}_chunk_{args.chunk:02d}.jsonl"
    # Global scan prevents duplicate ratings when watchdog concurrency changes.
    done = all_existing_ids(args.rater)

    for eval_id in ids:
        if eval_id in done:
            continue
        try:
            result = call(api_key, args.model, eval_id)
            record = {
                **result,
                "rater": args.rater,
                "model": args.model,
                "correct_binding": result["left_choice"] == "A"
                and result["right_choice"] == "B",
                "A_to_B_leakage": result["right_choice"] == "A",
                "B_to_A_leakage": result["left_choice"] == "B",
            }
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "rated", "eval_id": eval_id}), flush=True)
        except Exception as exc:
            with errors.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"eval_id": eval_id, "error": type(exc).__name__, "message": str(exc)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                json.dumps(
                    {"event": "failed", "eval_id": eval_id, "message": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
