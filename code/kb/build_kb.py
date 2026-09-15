"""Build the cultural knowledge base (Stage 1, Task 2).

Per concept: parse (LLM) -> Wikidata resolve (guards) -> verify+clean (LLM) ->
LLM fallback for misses -> familiarity. Resumable JSONL; every record carries source.

Usage:
  python build_kb.py --concepts <concepts_all.jsonl> --out <kb_all.jsonl> \
      --model gpt-4o-mini --fallback-model gpt-4o [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import enrich  # noqa: E402
from resolve import resolve_concept  # noqa: E402


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    return done


def build_one(c: dict, model: str, fallback_model: str) -> dict:
    key, concept, culture, category = c["key"], c["concept"], c["culture"], c["category"]
    rec = {"key": key, "concept": concept, "culture": culture, "category": category,
           "source_dataset": c.get("source"), "meta": c.get("meta", {}),
           "model": model, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    t0 = time.time()
    try:
        p = enrich.parse_concept(concept, culture, category, model)
        rec["parse"] = p
        res = resolve_concept(concept, culture, category, object_type=p.get("object_type", ""),
                              query=p.get("wikidata_query", ""), max_queries=3, max_checks=2)
        rec["wikidata"] = {
            "qid": res.get("qid"), "label": res.get("label"), "description": res.get("description"),
            "url": res.get("url"), "sitelinks": res.get("sitelinks"),
            "facts": res.get("facts", {}), "used_query": res.get("used_query"),
            "rejects": res.get("rejects"),
        }
        if res.get("qid"):
            vc = enrich.verify_and_clean(concept, culture, category, res.get("label", ""),
                                         res.get("description", ""), res.get("facts", {}), model)
            rec["verify"] = vc
            if vc.get("match") and vc.get("knowledge_text"):
                rec["source"] = "wikidata"
                rec["knowledge_text"] = vc["knowledge_text"]
            else:
                rec["source"] = "llm_fallback"
                rec["knowledge_text"] = enrich.llm_fallback(concept, culture, category, fallback_model)
                rec["fallback_model"] = fallback_model
                rec["fallback_reason"] = vc.get("reason", "")
        else:
            rec["source"] = "llm_fallback"
            rec["knowledge_text"] = enrich.llm_fallback(concept, culture, category, fallback_model)
            rec["fallback_model"] = fallback_model
            rec["fallback_reason"] = "no_candidate"
        rec["familiarity"] = enrich.familiarity(concept, culture, model)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--fallback-model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    concepts = [json.loads(l) for l in args.concepts.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    n = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for c in concepts:
            if c["key"] in done:
                continue
            rec = build_one(c, args.model, args.fallback_model)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done.add(c["key"])
            n += 1
            print(json.dumps({"built": rec["key"], "concept": rec["concept"],
                              "source": rec.get("source"), "qid": rec.get("wikidata", {}).get("qid"),
                              "sec": rec.get("seconds")}, ensure_ascii=False), flush=True)
            if args.limit and n >= args.limit:
                break
    print(f"done {n} new; total {len(done)}")


if __name__ == "__main__":
    main()
