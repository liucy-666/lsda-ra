"""Split a JSONL manifest into N chunk files for parallel scoring."""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunks", type=int, default=6)
    args = ap.parse_args()

    lines = [l for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.chunks):
        chunk = lines[i :: args.chunks]
        path = args.out_dir / f"chunk_{i:02d}.jsonl"
        path.write_text("\n".join(chunk) + ("\n" if chunk else ""), encoding="utf-8")
        print(f"chunk {i:02d}: {len(chunk)} -> {path}")


if __name__ == "__main__":
    main()
