"""Build an auditable cultural knowledge base from normalized concepts.

The workstation path is model-free: ``--offline`` validates source rows and
writes unresolved records with provenance. A network/model machine can omit
``--offline`` and provide ``--cache`` and ``--llm`` to retrieve Wikidata facts
and optionally refine the short visual ``knowledge_text``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from .culture_trip import build_knowledge_text, culture_trip, format_information
    from .llm_client import OpenAICompatibleClient
    from .parse_entities import normalize_record, read_records
    from .wikidata_client import WikidataClient, claim_values
except ImportError:
    from culture_trip import build_knowledge_text, culture_trip, format_information
    from llm_client import OpenAICompatibleClient
    from parse_entities import normalize_record, read_records
    from wikidata_client import WikidataClient, claim_values


REJECT_TERMS = re.compile(
    r"\b(people|person|family|organization|company|school|university|museum|"
    r"painting|film|song|album|festival|event|city|village|river|mountain|"
    r"building|church|mosque|temple|landmark|house|sports? team)\b", re.I
)

# Demonym and object hints only guide Wikidata candidate ranking; they are not
# written back as asserted cultural facts.
DEMONYM = {
    "japanese": "Japan", "chinese": "China", "korean": "Korea", "indian": "India",
    "persian": "Iran", "russian": "Russia", "mexican": "Mexico", "moroccan": "Morocco",
    "vietnamese": "Vietnam", "indonesian": "Indonesia", "thai": "Thailand", "italian": "Italy",
    "french": "France", "spanish": "Spain", "romanian": "Romania", "bhutanese": "Bhutan",
    "ghanaian": "Ghana", "hungarian": "Hungary", "dutch": "Netherlands", "slovak": "Slovakia",
    "tunisian": "Tunisia", "ottoman": "Turkey", "armenian": "Armenia", "serbian": "Serbia",
    "burmese": "Myanmar", "georgian": "Georgia", "azerbaijani": "Azerbaijan", "tibetan": "Tibet",
    "turkmen": "Turkmenistan", "palestinian": "Palestine", "panamanian": "Panama",
    "ukrainian": "Ukraine", "turkish": "Turkey", "venetian": "Venice", "german": "Germany",
    "algerian": "Algeria", "polish": "Poland", "bulgarian": "Bulgaria", "peruvian": "Peru",
    "english": "England", "portuguese": "Portugal", "kashmiri": "Kashmir", "syrian": "Syria",
    "egyptian": "Egypt", "bosnian": "Bosnia", "cypriot": "Cyprus", "greek": "Greece",
    "nepalese": "Nepal", "benin": "Benin", "iban": "Borneo", "sumbanese": "Sumba",
    "timorese": "Timor", "lao": "Laos", "afghan": "Afghanistan",
}
TYPE_HINTS = {
    "carpet": ["carpet", "rug", "floor"], "rug": ["rug", "carpet"], "porcelain": ["porcelain"],
    "pottery": ["pottery", "ceramic"], "ceramic": ["ceramic", "pottery"], "vase": ["vase"],
    "jar": ["jar", "vessel"], "plate": ["plate", "dish"], "bowl": ["bowl"], "box": ["box"],
    "textile": ["textile", "fabric", "cloth"], "cloth": ["cloth", "textile", "fabric"],
    "panel": ["panel", "textile", "fabric"], "embroidery": ["embroidery"],
    "kimono": ["kimono", "garment", "robe"], "skirt": ["skirt", "garment"],
    "brocade": ["brocade", "textile", "silk"], "silk": ["silk", "textile"],
    "batik": ["batik", "textile"], "ikat": ["ikat", "textile"], "bronze": ["bronze", "metal"],
    "copper": ["copper", "metal"], "brass": ["brass", "metal"], "silver": ["silver", "metal"],
    "iron": ["iron", "metal"], "metal": ["metal"], "lacquer": ["lacquer"], "glass": ["glass"],
    "wood": ["wood", "wooden"], "enamel": ["enamel", "cloisonne"], "cloisonn": ["cloisonne", "enamel"],
    "leather": ["leather"], "samovar": ["samovar", "metal"],
}
ARTICLES = {"a", "an", "the"}
BAD_DESC = ("museum", "gallery", "fragment", "photograph", "painting", "collection", "auction",
            "study", "from the", "loan", "archive", "dish from", "tabletop", "of flowers", "inv.",
            "yale", "century", "archival")


def clean_query(noun: str) -> str:
    return " ".join(word for word in str(noun).replace(",", " ").split() if word.lower() not in ARTICLES)


def query_ladder(noun: str) -> list[str]:
    words = clean_query(noun).split()
    ladder = [" ".join(words)]
    if len(words) >= 3:
        ladder.extend((" ".join(words[:1] + words[-2:]), " ".join(words[:1] + words[-1:])))
        ladder.extend(f"{words[0]} {word}" for word in words[1:-1])
    ladder.extend(" ".join(words[start:]) for start in range(1, max(1, len(words) - 1)))
    seen: set[str] = set()
    output = []
    for query in ladder:
        normalized = query.lower()
        if query and normalized not in seen:
            seen.add(normalized)
            output.append(query)
    return output


def filter_hits(hits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(hit) for hit in hits]
    good = [row for row in rows if not any(term in str(row.get("description", "")).lower() for term in BAD_DESC)]
    return good or rows


def infer_hints(noun: str) -> tuple[str, list[str]]:
    low = str(noun).lower()
    country = next((value for key, value in DEMONYM.items() if key in low.split()), "")
    types = sorted({hint for key, hints in TYPE_HINTS.items() if key in low for hint in hints})
    return country, types


def _claim_label_ids(entity: Mapping[str, Any], props: Iterable[str]) -> list[str]:
    values: list[str] = []
    for prop in props:
        values.extend(str(value) for value in claim_values(entity, prop) if value)
    return values


def _candidate_score(record: Mapping[str, Any], result: Mapping[str, Any], *, country_hint: str = "", type_hints: Iterable[str] = ()) -> float:
    concept = str(record.get("concept", "")).lower()
    label = str(result.get("label", "")).lower()
    desc = str(result.get("description", "")).lower()
    score = 10.0 if concept and concept == label else 0.0
    if concept and concept in label:
        score += 3.0
    culture = str(record.get("culture", "")).lower()
    if culture and culture in (label + " " + desc):
        score += 2.0
    score += sum(1.5 for hint in type_hints if str(hint).lower() in (label + " " + desc))
    if country_hint and str(country_hint).lower() in (label + " " + desc):
        score += 2.0
    if REJECT_TERMS.search(label + " " + desc):
        score -= 10.0
    return score


def resolve_candidate(
    record: Mapping[str, Any], candidates: Iterable[Mapping[str, Any]], *,
    country_hint: str = "", type_hints: Iterable[str] = (),
) -> Optional[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: (_candidate_score(record, item, country_hint=country_hint, type_hints=type_hints), str(item.get("id", item.get("qid", "")))), reverse=True)
    if not ranked or _candidate_score(record, ranked[0], country_hint=country_hint, type_hints=type_hints) < 0:
        return None
    return dict(ranked[0])


def _fact_map(
    record: Mapping[str, Any], entity: Mapping[str, Any], *, client: Optional[Any] = None,
    language: str = "en",
) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "label": entity.get("labels", {}).get("en", {}).get("value", record.get("concept")),
        "description": entity.get("descriptions", {}).get("en", {}).get("value", ""),
    }
    aliases = entity.get("aliases", {}).get("en", [])
    if aliases:
        facts["aliases"] = [item.get("value") for item in aliases if isinstance(item, Mapping)]
    claim_props = {
        "country_of_origin": "P495", "country": "P17", "material_used": "P186",
        "subclass_of": "P279", "genre": "P136", "depicts": "P180", "culture": "P2596",
    }
    claim_ids = {name: _claim_label_ids(entity, [prop]) for name, prop in claim_props.items()}
    labels: dict[str, str] = {}
    if client and hasattr(client, "get_entities"):
        related = sorted({qid for values in claim_ids.values() for qid in values if qid.startswith("Q")})
        try:
            linked = client.get_entities(related, language=language)
            for qid, value in linked.items():
                label = value.get("labels", {}).get(language, {}).get("value") or value.get("labels", {}).get("en", {}).get("value")
                if label:
                    labels[qid] = str(label)
        except Exception:
            pass
    for name, values in claim_ids.items():
        values = [labels.get(value, value) for value in values]
        if values:
            facts[name] = values
    return facts


def _culture_matches(culture: str, labels: Iterable[str]) -> bool:
    aliases = {"cn": "china", "in": "india", "jp": "japan", "kr": "korea",
               "ir": "iran", "tr": "turkey", "mx": "mexico", "pe": "peru",
               "uz": "uzbekistan", "gh": "ghana", "id": "indonesia"}
    wanted = aliases.get(culture.strip().lower(), culture.strip().lower())
    return any(wanted and wanted in str(label).lower() for label in labels)


def build_entry(
    record: Mapping[str, Any], language: str = "en", *, client: Optional[Any] = None,
    llm: Optional[Any] = None, online: bool = True,
) -> dict[str, Any]:
    """Resolve one row while retaining source fields and audit provenance."""
    row = dict(record)
    if row.get("culture_noun") and not row.get("concept"):
        row["concept"] = row["culture_noun"]
    base = normalize_record(row, source=str(row.get("source", "unknown")), llm=llm)
    result: Optional[Mapping[str, Any]] = None
    entity: Optional[Mapping[str, Any]] = None
    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    if base.get("qid") and client and online:
        entity = client.get_entity(base["qid"], language=language)
        result = {"id": base["qid"], "label": base["concept"], "description": ""} if entity else None
    elif client and online and base.get("concept"):
        try:
            country_hint, type_hints = infer_hints(base["concept"])
            seen: set[str] = set()
            for query in query_ladder(base["concept"]):
                hits = filter_hits(client.search(query, language=language, limit=10))
                for item in hits:
                    identifier = str(item.get("id") or item.get("qid") or "")
                    if identifier and identifier not in seen:
                        seen.add(identifier)
                        candidates.append(item)
                if len(candidates) >= 6:
                    break
            result = resolve_candidate(base, candidates, country_hint=country_hint, type_hints=type_hints)
            if result:
                entity = client.get_entity(str(result.get("id") or result.get("qid")), language=language)
        except Exception as exc:
            warnings.append(type(exc).__name__)
    if not entity:
        warnings.append("unresolved")
    facts = _fact_map(base, entity, client=client, language=language) if entity else {}
    if base.get("culture") and entity and client and hasattr(client, "get_entities"):
        claim_ids = _claim_label_ids(entity, ["P495", "P17"])
        try:
            linked = client.get_entities(claim_ids, language=language)
            labels = [value.get("labels", {}).get("en", {}).get("value", "")
                      for value in linked.values() if isinstance(value, Mapping)]
            labels = [label for label in labels if label]
            if labels:
                facts["country_claim_labels"] = labels
                if not _culture_matches(str(base["culture"]), labels):
                    warnings.append("culture_claim_mismatch")
        except Exception:
            warnings.append("country_claim_unresolved")
    knowledge_text, origin = build_knowledge_text(facts, llm=llm)
    qid = (result or {}).get("id") or (result or {}).get("qid")
    out = dict(base)
    out.update({
        "wikidata": {"qid": qid, "label": (result or {}).get("label"),
                     "url": "https://www.wikidata.org/wiki/" + str(qid) if qid else None,
                     "facts": facts},
        "candidates": candidates,
        "knowledge_text": knowledge_text,
        "origin": "llm_refined" if origin == "llm_refined" else ("wikidata" if entity else "unresolved"),
        "status": "ok" if entity and "culture_claim_mismatch" not in warnings else ("warning" if entity else "unresolved"),
        "warnings": sorted(set(warnings)),
    })
    return out


def coverage_report(rows: list[Mapping[str, Any]]) -> str:
    counts = Counter(str(row.get("status", "unknown")) for row in rows)
    origins = Counter(str(row.get("origin", "unknown")) for row in rows)
    categories = Counter(str(row.get("category", "unknown")) for row in rows)
    lines = ["# Cultural knowledge-base coverage", "", f"Total concepts: **{len(rows)}**", "",
             "| status | count |", "|---|---:|"]
    lines.extend(f"| `{key}` | {counts[key]} |" for key in sorted(counts))
    lines.extend(["", "| origin | count |", "|---|---:|"])
    lines.extend(f"| `{key}` | {origins[key]} |" for key in sorted(origins))
    lines.extend(["", "## Category counts", "", "| category | count |", "|---|---:|"])
    lines.extend(f"| `{key}` | {categories[key]} |" for key in sorted(categories))
    unresolved = [row for row in rows if row.get("status") == "unresolved"]
    if unresolved:
        lines.extend(["", "## Unresolved concepts", ""])
        lines.extend(f"- {row.get('concept', '')} ({row.get('culture', '')})" for row in unresolved)
    return "\n".join(lines) + "\n"


def _load_entities(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("--pairs-json/--entities must contain a JSON list")
    if all(isinstance(row, Mapping) and ("concept" in row or "culture_noun" in row) for row in raw):
        return [dict(row) for row in raw]
    output = []
    for index, pair in enumerate(raw, start=1):
        if not isinstance(pair, Mapping):
            continue
        values = [(key, str(pair[key])) for key in ("文化物体A", "A", "a", "文化物体B", "B", "b") if pair.get(key)]
        if len(values) < 2:
            values = [(str(key), str(value)) for key, value in pair.items() if isinstance(value, str)][:2]
        for side, (_, noun) in zip(("A", "B"), values[:2]):
            output.append({"key": f"p{index:03d}_{side}", "culture_noun": noun, "source": "pairs-json"})
    return output


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--concepts", type=Path)
    source.add_argument("--pairs-json", type=Path)
    source.add_argument("--entities", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--language", default="en")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    records = read_records(args.concepts) if args.concepts else _load_entities(args.pairs_json or args.entities)
    if args.limit:
        records = records[: args.limit]
    llm = OpenAICompatibleClient(base_url=args.llm_base_url, model=args.llm_model) if args.llm else None
    client = None if args.offline else WikidataClient(endpoint=args.endpoint or "https://www.wikidata.org/w/api.php", cache_path=args.cache)
    existing: dict[str, dict[str, Any]] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["key"])] = row
    pending = [row for row in records if str(row.get("key") or row.get("concept") or row.get("culture_noun")) not in existing]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8", newline="\n") as handle:
        for record in pending:
            entry = build_entry(record, args.language, client=client, llm=llm, online=not args.offline)
            if args.refine and entry.get("wikidata", {}).get("qid"):
                entry["refined"] = culture_trip(entry.get("concept", ""), entry.get("concept", ""), format_information(entry),
                                                 threshold=args.threshold, model=args.model)
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"built": entry["key"], "concept": entry.get("concept"), "qid": entry["wikidata"].get("qid")}, ensure_ascii=False), flush=True)
    if args.coverage_report:
        all_rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines() if line.strip()]
        args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_report.write_text(coverage_report(all_rows), encoding="utf-8")
    print(json.dumps({"records": len(records), "new": len(pending), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
