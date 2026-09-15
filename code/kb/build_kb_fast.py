"""Fast batched KB builder (Stage 1, Task 2).

Phase A: batch-resolve concepts by exact enwiki title + CUBE direct QIDs; batch facts.
Phase B: (optional) per-concept search fallback -- off by default (slow, rate-limited).
Phase C: batched LLM verify+clean, batched fallback, batched familiarity (20-25/request).
Resumable JSONL. Every record carries `source` = wikidata|llm_fallback.
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
import wikidata_client as wd  # noqa: E402
from resolve import resolve_concept, consistency  # noqa: E402

CHUNK = 20


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    return done


def title_variants(concept: str) -> list[str]:
    c = concept.strip()
    vs = [c, c[:1].upper() + c[1:], c.title()]
    seen, out = set(), []
    for v in vs:
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--fallback-model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--search-fallback", action="store_true")
    args = ap.parse_args()

    concepts = [json.loads(l) for l in args.concepts.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = load_done(args.out)
    todo = [c for c in concepts if c["key"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"todo {len(todo)} / total {len(concepts)}", flush=True)

    # ---- Phase A ----
    direct_qids, all_titles = [], []
    for c in todo:
        cid = str(c.get("meta", {}).get("cube_id", "") or "")
        if cid.startswith("Q"):
            direct_qids.append(cid)
        for t in title_variants(c["concept"]):
            all_titles.append(t)
    t0 = time.time()
    resolved = wd.get_by_titles(all_titles)
    entities = dict(resolved["entities"])
    for i in range(0, len(set(direct_qids)), 50):
        entities.update(wd.get_entities(sorted(set(direct_qids))[i : i + 50]))
    facts_by_qid = wd.batch_facts(entities)
    print(f"entities {len(entities)} (title {len(resolved['entities'])}, direct {len(set(direct_qids))}); "
          f"facts {len(facts_by_qid)} in {time.time()-t0:.0f}s", flush=True)

    # ---- build per-concept plan ----
    plan = []
    for c in todo:
        key = c["key"]
        qid = None
        cid = str(c.get("meta", {}).get("cube_id", "") or "")
        if cid.startswith("Q") and cid in facts_by_qid:
            qid = cid
        else:
            for t in title_variants(c["concept"]):
                if resolved["by_title"].get(t.lower()) in facts_by_qid:
                    qid = resolved["by_title"][t.lower()]
                    break
        facts = facts_by_qid.get(qid) if qid else None
        guard = None
        if facts:
            ok, reason = consistency(facts.get("facts", {}), facts.get("label", ""),
                                     facts.get("description", ""), c["culture"], "")
            if not ok:
                guard, facts = reason, None
        if not facts and args.search_fallback:
            res = resolve_concept(c["concept"], c["culture"], c["category"], max_queries=3, max_checks=2)
            if res.get("qid"):
                facts = {"qid": res["qid"], "label": res.get("label"), "description": res.get("description"),
                         "url": res.get("url"), "sitelinks": None, "facts": res.get("facts", {})}
        plan.append({"c": c, "facts": facts, "guard": guard})
    n_wd_planned = sum(1 for p in plan if p["facts"])
    print(f"planned wikidata {n_wd_planned} / {len(plan)}", flush=True)

    # ---- Phase C: batched LLM ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fh = args.out.open("a", encoding="utf-8")
    try:
        for start in range(0, len(plan), CHUNK):
            grp = plan[start : start + CHUNK]
            st = time.time()
            # verify+clean for those with facts
            payloads, pos = [], []
            for j, p in enumerate(grp):
                if p["facts"]:
                    payloads.append({"concept": p["c"]["concept"], "culture": p["c"]["culture"],
                                     "category": p["c"]["category"], "label": p["facts"].get("label", ""),
                                     "description": p["facts"].get("description", ""),
                                     "facts": p["facts"].get("facts", {})})
                    pos.append(j)
            vc = enrich.batch_verify_clean(payloads, args.model, chunk=CHUNK)
            vc_by_j = {pos[i]: vc.get(i, {}) for i in range(len(pos))}
            # fallback for no-facts or unverified
            fb_payloads, fb_pos = [], []
            for j, p in enumerate(grp):
                if (not p["facts"]) or (not vc_by_j.get(j, {}).get("match")) or (not vc_by_j.get(j, {}).get("knowledge_text")):
                    fb_payloads.append({"concept": p["c"]["concept"], "culture": p["c"]["culture"],
                                        "category": p["c"]["category"]})
                    fb_pos.append(j)
            fbres = enrich.batch_fallback(fb_payloads, args.fallback_model, chunk=CHUNK) if fb_payloads else {}
            fb_by_j = {fb_pos[i]: fbres.get(i, "") for i in range(len(fb_pos))}
            # familiarity
            fam_payloads = [{"concept": p["c"]["concept"], "culture": p["c"]["culture"]} for p in grp]
            fam = enrich.batch_familiarity(fam_payloads, args.model, chunk=25)

            for j, p in enumerate(grp):
                c = p["c"]
                rec = {"key": c["key"], "concept": c["concept"], "culture": c["culture"],
                       "category": c["category"], "source_dataset": c.get("source"),
                       "meta": c.get("meta", {}), "model": args.model,
                       "retrieved_at": datetime.now(timezone.utc).isoformat()}
                if p["facts"]:
                    rec["wikidata"] = p["facts"]
                if vc_by_j.get(j, {}).get("match") and vc_by_j.get(j, {}).get("knowledge_text"):
                    rec["source"] = "wikidata"
                    rec["knowledge_text"] = vc_by_j[j]["knowledge_text"]
                    rec["verify"] = {"match": True}
                else:
                    rec["source"] = "llm_fallback"
                    rec["knowledge_text"] = fb_by_j.get(j, "") or enrich.llm_fallback(
                        c["concept"], c["culture"], c["category"], args.fallback_model)
                    rec["fallback_model"] = args.fallback_model
                    rec["fallback_reason"] = (vc_by_j.get(j, {}) or {}).get("reason") or p.get("guard") or "no_candidate"
                rec["familiarity"] = fam.get(j, float("nan"))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps({"chunk": start // CHUNK, "done": min(start + CHUNK, len(plan)),
                              "sec": round(time.time() - st, 1)}), flush=True)
    finally:
        fh.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
