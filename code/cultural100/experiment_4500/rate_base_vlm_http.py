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


REQUIRED = ("standalone_A", "standalone_B", "candidate_A", "candidate_B", "shift", "structure")


def parse_payload(raw):
    if isinstance(raw, dict):
        value = raw
    else:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip(), flags=re.I | re.S)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return value if all(key in value for key in REQUIRED) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rater", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--tag", default="a_http_full")
    args = parser.parse_args()

    key = os.environ["TEST_API_KEY"]
    task = json.loads(args.task.read_text(encoding="utf-8"))
    rows = json.loads((args.base / "key" / "blind_map.json").read_text(encoding="utf-8"))
    lookup = {row["sample_id"]: row for row in rows}
    order = json.loads(args.order.read_text(encoding="utf-8"))
    out = args.base / "ratings" / args.rater / f"ratings_{args.rater}_{args.tag}_chunk_0.jsonl.raw"
    out.parent.mkdir(parents=True, exist_ok=True)
    schema = json.dumps(task["output_schema"], ensure_ascii=False, separators=(",", ":"))

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8-sig").splitlines():
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue
            if parse_payload(envelope.get("raw")) is not None:
                done.add(envelope.get("sample_id"))

    ok = failed = 0
    for index, sample_id in enumerate(order, 1):
        if sample_id in done:
            continue
        row = lookup[sample_id]
        prompt = (
            task["instructions"]
            + "\n\nSAMPLE ID: " + sample_id
            + "\nTARGET A: " + row["entity_A"]
            + "\nA DIAGNOSTIC: " + row["entity_A_diagnostic"]
            + "\nTARGET B: " + row["entity_B"]
            + "\nB DIAGNOSTIC: " + row["entity_B_diagnostic"]
            + "\n\nRequired JSON schema: " + schema
        )
        jpg = (args.base / "blind" / f"{sample_id}.jpg").read_bytes()
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
        body = {
            "model": args.model,
            "temperature": 0.1,
            "max_tokens": 3500,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
        }
        raw = None
        for attempt in range(2):
            request = urllib.request.Request(
                "https://api.openlux.ai/v1/chat/completions",
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    result = json.loads(response.read().decode("utf-8"))
                raw = result["choices"][0]["message"]["content"]
                if isinstance(raw, list):
                    raw = " ".join(map(str, raw))
                if parse_payload(raw) is not None:
                    break
                raw = None
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                raw = None
            if attempt == 0:
                time.sleep(5)
        if raw is None:
            failed += 1
        else:
            record = {
                "sample_id": sample_id,
                "rater_id": args.rater,
                "rater_model": args.model,
                "raw": raw,
                "request_status": "",
            }
            with out.open("a", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            ok += 1
        if index % 10 == 0:
            print(f"[{index}/{len(order)}] ok={ok} failed={failed}", flush=True)
    print(f"DONE rater={args.rater} ok={ok} failed={failed} out={out}", flush=True)


if __name__ == "__main__":
    main()
