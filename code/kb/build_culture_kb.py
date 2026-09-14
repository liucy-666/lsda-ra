"""Build a cultural knowledge base from Wikidata (CUBE-style structured KB).

Each entry resolves one culture noun to a Wikidata entity and stores its structured
cultural facts (material, depicts, country of origin, genre, subclass chain). Optional
``--refine`` runs the Culture-TRIP iterative refinement to produce a visual knowledge text.

Usage:
  python build_culture_kb.py --pairs-json <cultural_pairs_100.json> --out <kb.jsonl>
  python build_culture_kb.py --entities <entities.json> --out <kb.jsonl> --refine --model gpt-5.4
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import wikidata_client as wd  # noqa: E402

# Demonym -> country, for Wikidata disambiguation only.
DEMONYM = {
    "japanese": "Japan", "chinese": "China", "korean": "Korea", "indian": "India",
    "persian": "Iran", "russian": "Russia", "mexican": "Mexico", "moroccan": "Morocco",
    "vietnamese": "Vietnam", "indonesian": "Indonesia", "thai": "Thailand", "italian": "Italy",
    "french": "France", "spanish": "Spain", "romanian": "Romania", "bhutanese": "Bhutan",
    "ghanaian": "Ghana", "hungarian": "Hungary", "dutch": "Netherlands", "slovak": "Slovakia",
    "tunisian": "Tunisia", "ottoman": "Turkey", "armenian": "Armenia", "serbian": "Serbia",
    "burmese": "Myanmar", "georgian": "Georgia", "azerbaijani": "Azerbaijan", "tibetan": "Tibet",
    "turkmen": "Turkmenistan", "palestinian": "Palestine", "panamanian": "Panama",
    "ukrainian": "Ukraine", "turkish": "Turkey", "navajo": "Navajo", "venetian": "Venice",
    "german": "Germany", "algerian": "Algeria", "polish": "Poland", "bulgarian": "Bulgaria",
    "peruvian": "Peru", "english": "England", "portuguese": "Portugal", "kashmiri": "Kashmir",
    "syrian": "Syria", "egyptian": "Egypt", "bosnian": "Bosnia", "cypriot": "Cyprus",
    "greek": "Greece", "nepalese": "Nepal", "benin": "Benin", "iban": "Borneo",
    "sumbanese": "Sumba", "timorese": "Timor", "lao": "Laos", "afghan": "Afghanistan",
}

# Object-type keywords -> tokens expected in the Wikidata description.
TYPE_HINTS = {
    "carpet": ["carpet", "rug", "floor"], "rug": ["rug", "carpet"],
    "porcelain": ["porcelain"], "pottery": ["pottery", "ceramic", "ceramic"],
    "ceramic": ["ceramic", "pottery"], "vase": ["vase"], "jar": ["jar", "vessel"],
    "plate": ["plate", "dish"], "bowl": ["bowl"], "box": ["box"],
    "textile": ["textile", "fabric", "cloth"], "cloth": ["cloth", "textile", "fabric"],
    "panel": ["panel", "textile", "fabric"], "embroidery": ["embroidery"],
    "kimono": ["kimono", "garment", "robe"], "skirt": ["skirt", "garment"],
    "brocade": ["brocade", "textile", "silk"], "silk": ["silk", "textile"],
    "batik": ["batik", "textile"], "ikat": ["ikat", "textile"],
    "bronze": ["bronze", "metal"], "copper": ["copper", "metal"], "brass": ["brass", "metal"],
    "silver": ["silver", "metal"], "iron": ["iron", "metal"], "metal": ["metal"],
    "lacquer": ["lacquer"], "glass": ["glass"], "wood": ["wood", "wooden"],
    "enamel": ["enamel", "cloisonne"], "cloisonn": ["cloisonne", "enamel"],
    "leather": ["leather"], "samovar": ["samovar", "metal"],
}


ARTICLES = {"a", "an", "the"}
BAD_DESC = ("museum", "gallery", "fragment", "photograph", "painting", "collection",
            "auction", "study", "from the", "loan", "archive", "dish from",
            "tabletop", "of flowers", "inv.", "yale", "century", "archival")


def clean_query(noun: str) -> str:
    words = [w for w in noun.replace(",", " ").split() if w.lower() not in ARTICLES]
    return " ".join(words)


def query_ladder(noun: str) -> list[str]:
    """Wikidata search is prefix/fuzzy; generate candidate queries: the full cleaned phrase,
    demonym+object combinations, then progressively shorter suffixes."""
    words = clean_query(noun).split()
    ladder = [" ".join(words)]
    if len(words) >= 3:
        ladder.append(" ".join(words[:1] + words[-2:]))   # e.g. "Chinese porcelain vase"
        ladder.append(" ".join(words[:1] + words[-1:]))   # e.g. "Chinese vase"
        for w in words[1:-1]:
            ladder.append(f"{words[0]} {w}")              # e.g. "Chinese porcelain"
    for start in range(1, max(1, len(words) - 1)):
        ladder.append(" ".join(words[start:]))
    seen, out = set(), []
    for q in ladder:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def filter_hits(hits: list[dict]) -> list[dict]:
    good = [h for h in hits if not any(b in h["description"].lower() for b in BAD_DESC)]
    return good or hits


def infer_hints(noun: str) -> tuple[str, list[str]]:
    low = noun.lower()
    country = ""
    for demo, country_name in DEMONYM.items():
        if demo in low.split() or f" {demo} " in f" {low} ":
            country = country_name
            break
    types = []
    for kw, hints in TYPE_HINTS.items():
        if kw in low:
            types.extend(hints)
    return country, sorted(set(types))


def load_entities(args) -> list[dict]:
    if args.entities:
        data = json.loads(Path(args.entities).read_text(encoding="utf-8"))
        return [{"key": e.get("key", e["culture_noun"]), "culture_noun": e["culture_noun"]} for e in data]
    data = json.loads(Path(args.pairs_json).read_text(encoding="utf-8"))
    ents = []
    for i, p in enumerate(data, start=1):
        ks = list(p.keys())
        ents.append({"key": f"p{i:03d}_A", "culture_noun": p[ks[0]]})
        ents.append({"key": f"p{i:03d}_B", "culture_noun": p[ks[1]]})
    return ents


def build_entry(entity: dict, language: str = "en") -> dict:
    noun = entity["culture_noun"]
    country, types = infer_hints(noun)
    hits, used_query = [], ""
    seen = set()
    for query in query_ladder(noun):
        got = filter_hits(wd.search(query, limit=8, language=language))
        for h in got:
            if h["qid"] not in seen:
                seen.add(h["qid"])
                hits.append(h)
        if not used_query and got:
            used_query = query
        if len(hits) >= 6:  # enough candidates collected
            break
    best = wd.pick_best_hit(hits, noun, country, types)
    entry = {
        "key": entity["key"], "culture_noun": noun, "used_query": used_query,
        "country_hint": country, "type_hints": types,
        "candidates": hits, "wikidata": {}, "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    if best:
        entry["wikidata"] = wd.collect_facts(best["qid"], language)
        entry["wikidata"]["search_description"] = best["description"]
    return entry


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pairs-json", type=Path)
    g.add_argument("--entities", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int, default=0, help="max new entries (0 = all)")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--threshold", type=int, default=40)
    args = ap.parse_args()

    entities = load_entities(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])

    n_new = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for ent in entities:
            if ent["key"] in done:
                continue
            entry = build_entry(ent, args.language)
            if args.refine:
                from culture_trip import culture_trip, format_information
                info = format_information(entry)
                res = culture_trip(ent["culture_noun"], ent["culture_noun"], info,
                                   threshold=args.threshold, model=args.model)
                entry["refined"] = res
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            n_new += 1
            print(json.dumps({"built": ent["key"], "noun": ent["culture_noun"],
                              "qid": entry["wikidata"].get("qid"),
                              "label": entry["wikidata"].get("label")}), flush=True)
            if args.limit and n_new >= args.limit:
                break


if __name__ == "__main__":
    main()
