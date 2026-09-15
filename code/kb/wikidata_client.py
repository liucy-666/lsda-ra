"""Wikidata API access with deterministic JSONL caching.

The client speaks only the public search/entity endpoints. Cache paths are
caller-supplied so credentials and generated data never enter the repository.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

API = "https://www.wikidata.org/w/api.php"
DEFAULT_ENDPOINT = API
UA = {"User-Agent": "MMDIT-CultureKB/0.1 (research; contact: local)"}
MIN_INTERVAL = float(os.environ.get("WD_MIN_INTERVAL", "0.6"))
_LAST = [0.0]


class WikidataRequestError(RuntimeError):
    """Raised when Wikidata remains unavailable after retries."""


def _claim_value(claim: Mapping[str, Any]) -> Any:
    value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
    return value.get("id") if isinstance(value, Mapping) and "id" in value else value


class WikidataClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 20.0,
        retries: int = 3,
        user_agent: str = "lsda-ra-kb/1.0 (research)",
        cache_path: Optional[Union[os.PathLike, str]] = None,
        opener: Optional[Any] = None,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.user_agent = user_agent
        self.cache_path = Path(cache_path) if cache_path else None
        self._opener = opener or urllib.request.urlopen
        self._cache: dict[str, Any] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        assert self.cache_path is not None
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping) and row.get("key"):
                self._cache[str(row["key"])] = row.get("value")

    def _save_cache(self, key: str, value: Any) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "value": value}, ensure_ascii=False, sort_keys=True) + "\n")

    def request(self, params: Mapping[str, Any]) -> dict[str, Any]:
        query = {"format": "json", "formatversion": "2", **params}
        key = urllib.parse.urlencode(sorted((str(k), str(v)) for k, v in query.items()))
        if key in self._cache:
            return self._cache[key]
        url = self.endpoint + "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                with self._opener(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8", errors="replace"))
                if not isinstance(data, dict):
                    raise ValueError("Wikidata returned a non-object response")
                self._cache[key] = data
                self._save_cache(key, data)
                return data
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 8.0))
        raise WikidataRequestError("Wikidata request failed after retries") from last

    def search(self, term: str, *, language: str = "en", limit: int = 10) -> list[dict[str, Any]]:
        data = self.request({"action": "wbsearchentities", "search": term, "language": language,
                             "uselang": language, "type": "item", "limit": max(1, min(int(limit), 50))})
        return [dict(item) for item in data.get("search", []) if isinstance(item, Mapping)]

    def get_entities(self, qids: Iterable[str], *, language: str = "en") -> dict[str, dict[str, Any]]:
        ids = [str(q).strip() for q in qids if str(q).strip()]
        if not ids:
            return {}
        data = self.request({"action": "wbgetentities", "ids": "|".join(ids[:50]), "languages": language,
                             "props": "info|labels|descriptions|aliases|claims"})
        return {str(key): dict(value) for key, value in data.get("entities", {}).items()
                if isinstance(value, Mapping)}

    def get_entity(self, qid: str, *, language: str = "en") -> Optional[dict[str, Any]]:
        return self.get_entities([qid], language=language).get(str(qid))


def claim_values(entity: Mapping[str, Any], property_id: str) -> list[Any]:
    claims = entity.get("claims", {})
    rows = claims.get(property_id, []) if isinstance(claims, Mapping) else []
    return [_claim_value(row) for row in rows if isinstance(row, Mapping)]

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
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lsda-ra-kb/1.0 (research)"})
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
            "props": "claims|labels|descriptions|aliases|sitelinks", "languages": language,
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
        "sitelinks": len(ent.get("sitelinks", {}) or {}),
        "facts": facts,
    }


def get_by_titles(titles: list[str], site: str = "enwiki", language: str = "en") -> dict:
    """Batch-resolve exact site titles (up to 50/call) with redirects.

    Returns {"entities": {qid: entity}, "by_title": {normalized_title_lower: qid}}.
    """
    entities, by_title = {}, {}
    for i in range(0, len(titles), 50):
        batch = [t for t in titles[i : i + 50] if t]
        if not batch:
            continue
        d = _get({
            "action": "wbgetentities", "sites": site, "titles": "|".join(batch),
            "props": "claims|labels|descriptions|sitelinks", "languages": language,
            "redirects": "yes",
        })
        norm = {n["from"]: n["to"] for n in d.get("query", {}).get("normalized", [])}
        redir = {r["from"]: r["to"] for r in d.get("query", {}).get("redirects", [])}
        for qid, ent in d.get("entities", {}).items():
            entities[qid] = ent
            if "missing" in ent:
                continue
            title = (ent.get("sitelinks", {}).get(site, {}) or {}).get("title")
            for cand in (ent.get("labels", {}).get(language, {}) or {}).get("value", ""), title:
                if cand:
                    by_title.setdefault(cand.lower(), qid)
                    by_title.setdefault(norm.get(cand, cand).lower(), qid)
        for frm, to in {**norm, **redir}.items():
            by_title.setdefault(frm.lower(), by_title.get(to.lower(), ""))
    return {"entities": entities, "by_title": {k: v for k, v in by_title.items() if v}}


def raw_claims(ent: dict) -> dict:
    return {name: _claim_qids(ent, prop) for prop, name in PROPS.items() if _claim_qids(ent, prop)}


def batch_facts(entities: dict, language: str = "en") -> dict:
    """Build fact dicts for many entities, resolving related QIDs to labels in batched calls."""
    raw_by_qid, related = {}, set()
    for qid, ent in entities.items():
        if "missing" in ent:
            continue
        rc = raw_claims(ent)
        raw_by_qid[qid] = rc
        for vals in rc.values():
            related.update(q for q in vals if q.startswith("Q"))
    labels = {}
    related = sorted(related)
    for i in range(0, len(related), 40):
        for q, e in get_entities(related[i : i + 40], language).items():
            labels[q] = label_of(e, language)
    out = {}
    for qid, rc in raw_by_qid.items():
        ent = entities[qid]
        out[qid] = {
            "qid": qid, "label": label_of(ent, language),
            "description": (ent.get("descriptions", {}).get(language) or {}).get("value", ""),
            "url": f"https://www.wikidata.org/wiki/{qid}",
            "sitelinks": len(ent.get("sitelinks", {}) or {}),
            "facts": {name: [labels.get(q, q) for q in vals] for name, vals in rc.items()},
        }
    return out


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


__all__ = [
    "DEFAULT_ENDPOINT", "WikidataClient", "WikidataRequestError", "claim_values",
    "search", "get_entities", "collect_facts", "pick_best_hit",
]
