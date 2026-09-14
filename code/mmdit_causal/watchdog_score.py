"""Watchdog: keep a JSONL manifest fully scored, resuming missing ids with retries.

Loops until every id in --manifest appears in <out-dir>/<prefix>_NN.jsonl (or --max-rounds
reached). Each round rebuilds a "missing" manifest, splits it into --chunks, and runs
--workers parallel score_images.py processes; dead/incomplete runs are retried.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCORER = Path(__file__).resolve().parent / "score_images.py"


def load_records(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_ids(paths):
    ids = set()
    for p in paths:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        ids.add(json.loads(line)["id"])
                    except json.JSONDecodeError:
                        pass
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--raters", default="GPT54,GEMINI35")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--max-rounds", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.manifest)
    by_id = {r["id"]: r for r in records}
    print(f"[watchdog] manifest={len(records)} ids", flush=True)

    for round_index in range(args.max_rounds):
        out_files = sorted(args.out_dir.glob(f"{args.prefix}_*.jsonl"))
        done = load_ids(out_files) & set(by_id)
        missing = [rid for rid in by_id if rid not in done]
        print(f"[watchdog] round {round_index}: done={len(done)}/{len(records)} missing={len(missing)}", flush=True)
        if not missing:
            print("[watchdog] ALL DONE", flush=True)
            return
        miss_dir = args.out_dir / "missing"
        miss_dir.mkdir(parents=True, exist_ok=True)
        procs = []
        chunk_files = []
        for i in range(args.chunks):
            chunk = missing[i :: args.chunks]
            mpath = miss_dir / f"{args.prefix}_miss_{i:02d}.jsonl"
            mpath.write_text("\n".join(json.dumps(by_id[r], ensure_ascii=False) for r in chunk) + ("\n" if chunk else ""), encoding="utf-8")
            chunk_files.append(mpath)
        for i in range(0, len(chunk_files), args.workers):
            batch = chunk_files[i : i + args.workers]
            running = []
            for mpath in batch:
                out_path = args.out_dir / f"{args.prefix}_r{round_index}_{mpath.stem}.jsonl"
                cmd = [sys.executable, str(SCORER), "--manifest", str(mpath), "--out", str(out_path),
                       "--raters", args.raters, "--sleep", str(args.sleep)]
                running.append((subprocess.Popen(cmd), out_path))
                time.sleep(1.0)
            for proc, out_path in running:
                proc.wait()
        time.sleep(2.0)
    print("[watchdog] max rounds reached; still incomplete", flush=True)


if __name__ == "__main__":
    main()
