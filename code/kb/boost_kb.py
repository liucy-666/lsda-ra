"""Boost KB wiki coverage via native-language Wikipedia sitelinks (zh/de/ko/es).

TU concepts were extracted from native Wikipedia; their native titles resolve to real Wikidata
entities that enwiki-title matching misses. This pass upgrades `llm_fallback` records to
`wikidata` when a native entity is found and passes country/non-object guards.
Writes kb_all_v2.jsonl (does not modify kb_all.jsonl).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import enrich  # noqa: E402
import wikidata_client as wd  # noqa: E402
from resolve import consistency  # noqa: E402

LANG_SITE = {"chinese": ("zhwiki", "zh"), "german": ("dewiki", "de"),
             "korean": ("kowiki", "ko"), "spanish": ("eswiki", "es")}


def facts_text(facts: dict) -> str:
    keep = [("material_used", "material"), ("depicts", "depicts"), ("genre", "genre"),
            ("subclass_of", "type"), ("instance_of", "type"), ("country_of_origin", "origin"),
            ("culture", "culture"), ("has_use", "use")]
    parts = []
    for k, name in keep:
        if facts.get(k):
            parts.append(f"{name}: {', '.join(map(str, facts[k][:4]))}")
    return "; ".join(parts)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.kb.read_text(encoding="utf-8").splitlines() if l.strip()]
    concepts = {json.loads(l)["key"]: json.loads(l)
                for l in args.concepts.read_text(encoding="utf-8").splitlines() if l.strip()}
    native = list(csv.DictReader(args.native.open(encoding="utf-8")))

    # map TU key -> native title (align by index where country+semantic_field match translated order)
    # concepts_all TU order equals translated csv order; native[i] aligns when country+field match.
    tu_keys = sorted(k for k, c in concepts.items() if c.get("source") == "TU")
    key_title = {}
    for i, k in enumerate(tu_keys):
        if i >= len(native):
            break
        n = native[i]
        c = concepts[k]
        if n["country"].strip().lower() == c["culture"].strip().lower() and \
           n["semantic_field"].strip().lower() == (c.get("meta", {}).get("semantic_field", "").strip().lower()):
            key_title[k] = (n["concept"], n["language"].strip().lower())
    print(f"native titles mapped: {len(key_title)} / {len(tu_keys)}", flush=True)

    # batch resolve per language
    qid_by_key = {}
    for lang, (site, code) in LANG_SITE.items():
        keys = [k for k, (t, lg) in key_title.items() if lg == lang]
        titles = [key_title[k][0] for k in keys]
        if not titles:
            continue
        res = wd.get_by_titles(titles, site=site)
        for k in keys:
            t = key_title[k][0]
            q = res["by_title"].get(t.lower()) or res["by_title"].get(t)
            if q:
                qid_by_key[k] = q
        print(f"{lang}: {len(keys)} titles -> {sum(1 for k in keys if k in qid_by_key)} qids", flush=True)

    # fetch facts for all newly found qids
    new_qids = sorted({q for q in qid_by_key.values()})
    ents = {}
    for i in range(0, len(new_qids), 50):
        ents.update(wd.get_entities(new_qids[i : i + 50]))
    facts = wd.batch_facts(ents)
    print(f"facts for {len(facts)} new entities", flush=True)

    # LLM verify+clean for upgraded records
    upgrades = []
    for r in rows:
        if r["key"] in qid_by_key and r.get("source") != "wikidata":
            q = qid_by_key[r["key"]]
            f = facts.get(q)
            if not f:
                continue
            ok, reason = consistency(f.get("facts", {}), f.get("label", ""), f.get("description", ""),
                                     r["culture"], "")
            if ok:
                upgrades.append((r, f))
    print(f"guard-passed upgrades: {len(upgrades)}", flush=True)
    vc = enrich.batch_verify_clean([{"concept": r["concept"], "culture": r["culture"], "category": r["category"],
                                     "label": f.get("label", ""), "description": f.get("description", ""),
                                     "facts": f.get("facts", {})} for r, f in upgrades], args.model)

    up_by_key = {}
    for i, (r, f) in enumerate(upgrades):
        v = vc.get(i, {})
        kt = v.get("knowledge_text") if v.get("match") else ""
        text_kind = "llm_clean" if kt else "facts"
        if not kt:
            kt = facts_text(f.get("facts", {})) or r.get("knowledge_text", "")
        up_by_key[r["key"]] = {"wikidata": f, "knowledge_text": kt, "text_kind": text_kind,
                               "retrieval": "native_wiki", "verify": {"match": bool(v.get("match"))}}

    # write v2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            if r["key"] in up_by_key:
                r = {**r, **up_by_key[r["key"]], "source": "wikidata"}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    import collections
    c = collections.Counter(r.get("source") for r in
                            [({**r, **up_by_key[r["key"]], "source": "wikidata"} if r["key"] in up_by_key else r) for r in rows])
    n = len(rows)
    print("sources:", dict(c), "wiki_rate:", round(100 * c.get("wikidata", 0) / n, 1), "%")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
