"""Download evaluation model weights into D:\\Python\\MMDIT\\models (or --out).

Models (complete set for the unified automated evaluation):
  - facebook/dinov2-large          (perceptual / texture-structure similarity)
  - openai/clip-vit-large-patch14  (CLIP-I / CLIP-T reference similarity)
  - facebook/sam-vit-base          (instance masks recomputed locally)
  - google/siglip2-so400m-patch14-naflex  (MaSC-style masked-maxcos backbone)
  - Salesforce/blip-vqa-base       (T2I-CompBench style disentangled VQA)

Falls back to HF_ENDPOINT=https://hf-mirror.com on direct-connect failure.
No credentials are written anywhere.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

REPOS = [
    "facebook/dinov2-large",
    "openai/clip-vit-large-patch14",
    "facebook/sam-vit-base",
    "google/siglip2-so400m-patch16-naflex",  # SigLIP2 SO400M NaFlex (patch16)
    "Salesforce/blip-vqa-base",
]


def download(repo_id: str, out_root: Path, mirror: bool) -> Path:
    from huggingface_hub import snapshot_download

    if mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    else:
        os.environ.pop("HF_ENDPOINT", None)
    name = repo_id.replace("/", "__")
    target = out_root / name
    t0 = time.time()
    print(f"[download] {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target), max_workers=4)
    print(f"[download] {repo_id} done in {time.time() - t0:.1f}s", flush=True)
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(r"D:\Python\MMDIT\models"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    for repo in REPOS:
        try:
            download(repo, args.out, mirror=False)
            ok.append(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[download] direct failed for {repo}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            try:
                download(repo, args.out, mirror=True)
                ok.append(repo)
            except Exception as exc2:  # noqa: BLE001
                print(f"[download] mirror also failed for {repo}: {type(exc2).__name__}: {str(exc2)[:200]}", flush=True)
                fail.append(repo)
    print(f"OK: {ok}", flush=True)
    print(f"FAILED: {fail}", flush=True)
    if fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
