"""Resolve a cultural concept to a Wikidata entity and collect structured facts.

Disambiguation guards: verify candidate country and object type before accepting; reject
non-object artifacts (museums, teams, villages, fragments...). Returns qid=None when no
consistent candidate exists, so the caller can fall back to LLM-generated knowledge.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wikidata_client as wd  # noqa: E402
from build_culture_kb import query_ladder, filter_hits, TYPE_HINTS, BAD_DESC  # noqa: E402

COUNTRY_SYN = {
    "south korea": ["korea", "republic of korea", "korean"],
    "china": ["people's republic of china", "chinese", "prc"],
    "united states": ["usa", "united states of america", "u.s.", "american"],
    "united kingdom": ["uk", "england", "british", "great britain"],
    "italy": ["italian"], "france": ["french"], "japan": ["japanese"], "india": ["indian"],
    "brazil": ["brazilian"], "nigeria": ["nigerian"], "turkey": ["turkish", "türkiye"],
    "germany": ["german"], "spain": ["spanish", "españa"],
}
NON_OBJECT = (
    "team", "village", "municipality", "museum", "person", "company", "organization",
    "gallery", "university", "band", "film", "album", "building", "station", "genus",
    "species", "city", "town", "district", "province", "river", "mountain", "regency",
    "kampung", "census", "settlement", "airport", "railway",
)


def type_hints_for(object_type: str) -> list[str]:
    ot = (object_type or "").lower()
    hints = []
    for kw, hs in TYPE_HINTS.items():
        if kw in ot:
            hints.extend(hs)
    if ot:
        hints.append(ot)
    return sorted(set(hints))


def country_match(countries, culture: str) -> bool:
    if not countries:
        return False
    culture_l = culture.lower()
    ok_terms = {culture_l} | set(COUNTRY_SYN.get(culture_l, []))
    for c in countries:
        cl = str(c).lower()
        if any(t in cl or cl in t for t in ok_terms):
            return True
    return False


def consistency(facts: dict, label: str, description: str, culture: str, object_type: str):
    """Return (ok, reason). Hard rejects only: bad description, non-object, country mismatch."""
    text = f"{label} {description}".lower()
    inst = " ".join(map(str, facts.get("instance_of", []) + facts.get("subclass_of", []))).lower()
    countries = list(facts.get("country", [])) + list(facts.get("country_of_origin", []))
    if any(b in text for b in BAD_DESC):
        return False, "bad_description"
    if any(tok in inst for tok in NON_OBJECT) or any(tok in text for tok in NON_OBJECT):
        return False, "non_object"
    if countries and not country_match(countries, culture):
        return False, "country_mismatch"
    return True, "ok"


def _candidate_queries(concept: str, culture: str, query: str) -> list[str]:
    qs = []
    for q in [query, f"{concept} {culture}", concept, *query_ladder(concept)]:
        q = (q or "").strip()
        if q and q.lower() not in {x.lower() for x in qs}:
            qs.append(q)
    return qs


def resolve_concept(concept: str, culture: str, category: str, object_type: str = "",
                    query: str = "", max_queries: int = 4, max_checks: int = 3) -> dict:
    queries = _candidate_queries(concept, culture, query)
    hits, used = [], ""
    for q in queries[:max_queries]:
        got = filter_hits(wd.search(q, limit=8))
        have = {h["qid"] for h in hits}
        hits += [h for h in got if h["qid"] not in have]
        if not used and got:
            used = q
        if len(hits) >= 8:
            break

    ranked = sorted(hits, key=lambda h: -(
        (0.5 * sum(1 for t in concept.lower().split() if len(t) > 3 and t in h["label"].lower()))
        + (2.0 if culture.lower() in f"{h['label']} {h['description']}".lower() else 0.0)
        + (1.0 if object_type and object_type.lower() in h["description"].lower() else 0.0)
    ))
    rejects = []
    for cand in ranked[:max_checks]:
        facts = wd.collect_facts(cand["qid"])
        ok, reason = consistency(facts.get("facts", {}), facts.get("label", ""),
                                 facts.get("description", "") or cand["description"], culture, object_type)
        if ok:
            return {
                "qid": cand["qid"], "label": facts.get("label"), "description": facts.get("description"),
                "search_description": cand["description"], "used_query": used, "candidates": hits,
                "facts": facts.get("facts", {}), "url": facts.get("url"), "consistent": True,
            }
        rejects.append({"qid": cand["qid"], "label": facts.get("label"), "reason": reason})
    return {"qid": None, "used_query": used, "candidates": hits, "facts": {},
            "consistent": False, "rejects": rejects}
