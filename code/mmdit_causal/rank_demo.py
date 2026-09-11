"""Rank seeds by repair magnitude for demo selection."""
import argparse
import json
import re
from pathlib import Path

ARMS = ["baseline", "w_fix", "v_fix", "both_fix"]


def binding_of(row) -> float | None:
    if not isinstance(row, dict):
        return None
    v = row.get("right_is_b")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    a = row.get("right_is_a")
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return 1.0 - float(a)
    return None


def parse(path: Path):
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        m = re.match(r"e2_p(\d+)_s(\d+)_(.+)$", r["id"])
        if not m:
            continue
        seed, arm = int(m.group(2)), m.group(3)
        vals = [v for k, val in r.items() if k not in ("id", "image")
                for v in [binding_of(val)] if v is not None]
        rows.setdefault(seed, {})[arm] = (sum(vals) / len(vals)) if vals else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    merged = {}
    for p in args.scores:
        for seed, arms in parse(p).items():
            merged.setdefault(seed, {}).update(arms)

    ranked = []
    for seed, arms in merged.items():
        if not all(arms.get(a) is not None for a in ARMS):
            continue
        ranked.append({
            "seed": seed,
            "baseline": arms["baseline"],
            "both_fix": arms["both_fix"],
            "v_fix": arms["v_fix"],
            "w_fix": arms["w_fix"],
            "gain_both": arms["both_fix"] - arms["baseline"],
            "gain_v": arms["v_fix"] - arms["baseline"],
        })
    ranked.sort(key=lambda r: -r["gain_both"])
    for r in ranked:
        print(f"seed {r['seed']}: base={r['baseline']:.2f} w={r['w_fix']:.2f} v={r['v_fix']:.2f} both={r['both_fix']:.2f} | gain_both={r['gain_both']:+.2f} gain_v={r['gain_v']:+.2f}")


if __name__ == "__main__":
    main()
