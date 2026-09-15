"""Smoke test for LLM enrichment (parse -> Wikidata resolve -> clean -> familiarity).

Usage: python smoke_test.py --out <path.json> [--model gpt-4o-mini]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import enrich  # noqa: E402
from resolve import resolve_concept  # noqa: E402

CASES = [
    {"concept": "Ras malai", "culture": "India", "category": "food"},
    {"concept": "yugwa", "culture": "South Korea", "category": "food"},
    {"concept": "niu jiao hu", "culture": "China", "category": "food"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    results = []
    for c in CASES:
        t0 = time.time()
        rec = {"input": c, "model": args.model}
        try:
            rec["parse"] = enrich.parse_concept(c["concept"], c["culture"], c["category"], args.model)
            q = rec["parse"].get("wikidata_query", "")
            ot = rec["parse"].get("object_type", "")
            res = resolve_concept(c["concept"], c["culture"], c["category"], object_type=ot, query=q)
            rec["wikidata"] = {
                "qid": res.get("qid"), "label": res.get("label"),
                "description": res.get("description"), "used_query": res.get("used_query"),
                "facts": res.get("facts", {}), "rejects": res.get("rejects"),
            }
            if res.get("qid"):
                vc = enrich.verify_and_clean(c["concept"], c["culture"], c["category"],
                                             rec["wikidata"]["label"], rec["wikidata"]["description"],
                                             rec["wikidata"]["facts"], args.model)
                rec["verify"] = vc
                if vc["match"] and vc["knowledge_text"]:
                    rec["source"] = "wikidata"
                    rec["clean"] = vc["knowledge_text"]
                else:
                    rec["source"] = "llm_fallback"
                    rec["clean"] = enrich.llm_fallback(c["concept"], c["culture"], c["category"], "gpt-4o")
            else:
                rec["source"] = "llm_fallback"
                rec["clean"] = enrich.llm_fallback(c["concept"], c["culture"], c["category"], "gpt-4o")
            rec["familiarity"] = enrich.familiarity(c["concept"], c["culture"], args.model)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["seconds"] = round(time.time() - t0, 1)
        results.append(rec)
        print(json.dumps(rec, ensure_ascii=True, indent=1))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", args.out)


if __name__ == "__main__":
    main()
