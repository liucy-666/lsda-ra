"""Sample compositional cultural pairs (Stage 1, Task 1).

Rules: cross-culture + same category; each pair = one common + one rare concept
(rank-based within category), so binding + knowledge-absence are both exercised.
Deterministic (seed), records provenance for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kb"))
from llm_client import chat, parse_json  # noqa: E402

FILTER_PROMPT = """For EACH numbered item, decide if it is a SINGLE concrete physical artifact that
can be drawn realistically (e.g., a garment, vessel, instrument, textile, carved/cast craft object,
dish, food). Set ok=false for performances, dances, festivals, ceremonies, games, sports, music
genres/pieces, people, places, abstract concepts, or anything without a distinct visual form.
Return ONLY json: {{"results":[{{"i":0,"ok":true}}, ...]}}

ITEMS:
{items}
ANSWER:"""


def renderable_filter(items: list[dict], model: str = "gpt-4o-mini", chunk: int = 25) -> list[dict]:
    out = []
    for s in range(0, len(items), chunk):
        grp = items[s : s + chunk]
        lines = [f"#{i} {r['concept']} ({r['category']}, {r['culture']})" for i, r in enumerate(grp)]
        keep = {i: True for i in range(len(grp))}
        try:
            txt = chat(FILTER_PROMPT.format(items="\n".join(lines)), model=model, json_mode=True, max_tokens=900)
            d = parse_json(txt)
            keep = {int(r["i"]): bool(r.get("ok")) for r in d.get("results", [])}
        except Exception:
            pass
        out += [r for i, r in enumerate(grp) if keep.get(i, True)]
    return out


OBJECT_CATS = ("food", "art", "clothing", "music", "utensil")
NON_VISUAL = ("festival", "dance", "music", "performance", "ceremony", "game", "sport",
              "ritual", "parade", "theatre", "opera", "custom")
TEMPLATE = ("Neutral studio background: {A} on the left, {B} on the right; "
            "both fully visible, separate, and similar in size.")


def load_kb(path: Path) -> dict:
    kb = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            kb[r["key"]] = r
    return kb


def commonness(rec: dict, sitelink_norm: float) -> float:
    fam = rec.get("familiarity")
    fam = float(fam) if isinstance(fam, (int, float)) and fam == fam else 0.0
    sl = rec.get("wikidata", {}).get("sitelinks")
    sln = sitelink_norm if isinstance(sl, int) else fam
    return round(0.6 * fam + 0.4 * sln, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, default=0, help="pairs per category (0 = use quotas)")
    ap.add_argument("--quotas", default="food:600,art:150,clothing:100,music:100,utensil:50")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--rare-q", type=float, default=0.4, help="bottom fraction = rare pool")
    ap.add_argument("--common-q", type=float, default=0.6, help="top fraction = common pool")
    ap.add_argument("--llm-filter", action="store_true", help="LLM keep only renderable artifacts")
    ap.add_argument("--filter-model", default="gpt-4o-mini")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    kb = load_kb(args.kb)
    rows = [r for r in kb.values() if r.get("category") in OBJECT_CATS and r.get("knowledge_text")]
    max_sl = max([r.get("wikidata", {}).get("sitelinks") or 0 for r in rows] + [1])
    import math
    for r in rows:
        sl = r.get("wikidata", {}).get("sitelinks")
        r["_sl_norm"] = (math.log1p(sl) / math.log1p(max_sl)) if isinstance(sl, int) else None
        r["_common"] = commonness(r, r["_sl_norm"] if r["_sl_norm"] is not None else 0.0)

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    quotas = {}
    if args.per_category:
        quotas = {c: args.per_category for c in OBJECT_CATS}
    else:
        for kv in args.quotas.split(","):
            k, v = kv.split(":")
            quotas[k.strip()] = int(v)

    pairs = []
    for cat, items in by_cat.items():
        need = quotas.get(cat, 0)
        if need <= 0:
            continue
        if cat == "art":
            items = [r for r in items if not any(k in r["concept"].lower() for k in NON_VISUAL)]
        items = sorted(items, key=lambda r: r["_common"])
        n = len(items)
        if n < 2:
            continue
        rare_pool = items[: max(1, int(n * args.rare_q))]
        common_pool = items[int(n * (1 - args.common_q)) :]
        if args.llm_filter:
            keep = renderable_filter(common_pool + rare_pool, args.filter_model)
            keep_keys = {r["key"] for r in keep}
            common_pool = [r for r in common_pool if r["key"] in keep_keys]
            rare_pool = [r for r in rare_pool if r["key"] in keep_keys]
        rng.shuffle(rare_pool)
        rng.shuffle(common_pool)
        made, ci, ri = 0, 0, 0
        used_c, used_r = set(), set()
        while made < need and ci < len(common_pool) and ri < len(rare_pool):
            a, b = common_pool[ci], rare_pool[ri]
            if a["culture"] == b["culture"] or a["key"] in used_c or b["key"] in used_r or a["key"] == b["key"]:
                ri += 1
                if ri >= len(rare_pool):
                    ri = 0
                    ci += 1
                continue
            used_c.add(a["key"]); used_r.add(b["key"])
            pid = f"cmp_{len(pairs)+1:04d}"
            prompt = TEMPLATE.format(A=a["concept"], B=b["concept"])
            pairs.append({
                "pair_id": pid, "category": cat,
                "A": {"key": a["key"], "concept": a["concept"], "culture": a["culture"],
                      "commonness": a["_common"], "knowledge_text": a["knowledge_text"]},
                "B": {"key": b["key"], "concept": b["concept"], "culture": b["culture"],
                      "commonness": b["_common"], "knowledge_text": b["knowledge_text"]},
                "seeds": [42], "template": TEMPLATE, "prompt_ss": prompt,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            made += 1
            ci += 1
            ri += 1
        print(f"{cat}: {made}/{need} pairs  (items={n}, rare_pool={len(rare_pool)}, common_pool={len(common_pool)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"total pairs: {len(pairs)} -> {args.out}")


if __name__ == "__main__":
    main()
