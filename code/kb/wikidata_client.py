"""Minimal Wikidata client for cultural knowledge retrieval (urllib only).

The local shell cannot reach Wikipedia/Google, but ``www.wikidata.org`` is reachable.
Wikidata provides structured cultural facts (material used, depicts, country of origin,
genre, subclass chain) — the same structured-KB basis used by CUBE (DeepMind, NeurIPS 2024).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "MMDIT-CultureKB/0.1 (research; contact: local)"}
MIN_INTERVAL = 0.6
_LAST = [0.0]


def _sleep_gap():
    import time as _t
    gap = MIN_INTERVAL - (_t.time() - _LAST[0])
    if gap > 0:
        _t.sleep(gap)
    _LAST[0] = _t.time()

# Cultural / visual attributes kept from the Wikidata claims.
PROPS = {
    "P31": "instance_of",
    "P279": "subclass_of",
    "P495": "country_of_origin",
    "P17": "country",
    "P186": "material_used",
    "P180": "depicts",
    "P136": "genre",
    "P276": "location",
    "P361": "part_of",
    "P366": "has_use",
    "P2596": "culture",
    "P2341": "produced_by",
}


def _get(params: dict, retries: int = 5, timeout: int = 30) -> dict:
    params = dict(params)
    params.setdefault("format", "json")
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt, delay in enumerate((0, 2, 8, 20, 45)):
        if delay:
            time.sleep(delay)
        _sleep_gap()
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            last = f"HTTPError: {exc}"
            if exc.code == 429:
                time.sleep(5 * (attempt + 1))
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    raise RuntimeError(last)


def search(query: str, limit: int = 5, language: str = "en") -> list[dict]:
    d = _get({"action": "wbsearchentities", "search": query, "language": language, "limit": limit})
    out = []
    for hit in d.get("search", []):
        out.append({
            "qid": hit["id"],
            "label": hit.get("label", ""),
            "description": hit.get("description", ""),
            "aliases": hit.get("aliases", []),
            "match": hit.get("match", {}).get("text", ""),
        })
    return out


def get_entities(qids: list[str], language: str = "en") -> dict:
    out = {}
    for i in range(0, len(qids), 40):
        batch = [q for q in qids[i : i + 40] if q]
        if not batch:
            continue
        d = _get({
            "action": "wbgetentities", "ids": "|".join(batch),
            "props": "claims|labels|descriptions|aliases", "languages": language,
        })
        out.update(d.get("entities", {}))
    return out


def _claim_qids(entity: dict, prop: str) -> list[str]:
    vals = []
    for c in entity.get("claims", {}).get(prop, []):
        snak = c.get("mainsnak", {})
        if snak.get("snaktype") != "value":
            continue
        dv = snak.get("datavalue", {}).get("value")
        if isinstance(dv, dict) and "id" in dv:
            vals.append(dv["id"])
        elif isinstance(dv, str):
            vals.append(dv)
    seen, uniq = set(), []
    for v in vals:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def label_of(entity: dict, language: str = "en") -> str:
    return (entity.get("labels", {}).get(language) or {}).get("value") or entity.get("id", "")


def collect_facts(qid: str, language: str = "en") -> dict:
    """Fetch one entity and resolve its cultural claim values to English labels."""
    ent = get_entities([qid], language).get(qid)
    if not ent:
        return {"qid": qid, "error": "not found"}
    raw = {name: _claim_qids(ent, prop) for prop, name in PROPS.items()}
    raw = {k: v for k, v in raw.items() if v}
    related = sorted({q for vals in raw.values() for q in vals if q.startswith("Q")})
    labels = {r: label_of(rel, language) for r, rel in get_entities(related, language).items()}
    facts = {name: [labels.get(q, q) for q in vals] for name, vals in raw.items()}
    return {
        "qid": qid,
        "label": label_of(ent, language),
        "description": (ent.get("descriptions", {}).get(language) or {}).get("value", ""),
        "aliases": [a["value"] for a in ent.get("aliases", {}).get(language, [])],
        "url": f"https://www.wikidata.org/wiki/{qid}",
        "facts": facts,
    }


def pick_best_hit(hits: list[dict], culture_noun: str, country_hint: str = "", type_hints=None) -> dict | None:
    """Heuristic disambiguation: prefer label token overlap, then type/country keywords."""
    if not hits:
        return None
    type_hints = [t.lower() for t in (type_hints or [])]
    toks = {w for w in culture_noun.lower().replace(",", " ").split() if len(w) > 3}
    best, best_score = None, -1.0
    for h in hits:
        text = f"{h['label']} {h['description']}".lower()
        score = sum(0.5 for t in toks if t in text)
        score += sum(1.5 for t in type_hints if t in text)
        if country_hint and country_hint.lower() in text:
            score += 2.0
        if h["description"]:
            score += 0.3
        if score > best_score:
            best, best_score = h, score
    return best
