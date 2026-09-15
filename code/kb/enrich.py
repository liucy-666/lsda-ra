"""LLM enrichment for the cultural KB: concept parsing, knowledge-text cleaning, familiarity.

All calls go through llm_client (openlux). Default model is gpt-4o-mini; callers may pass
gpt-4o for hard cases (see PROTOCOL.md 3).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import chat, parse_json  # noqa: E402

PARSE_PROMPT = """You normalize a cultural object for a text-to-image knowledge base.

Return ONLY JSON:
{{
  "culture": "<country/culture this object belongs to>",
  "object_type": "<generic object type, e.g. vase, dish, textile, carpet, metalware>",
  "style": "<named style/technique/tradition, or empty>",
  "wikidata_query": "<a short 2-4 word English query best matching the CULTURAL CONCEPT for Wikidata lookup>",
  "notes": "<one short sentence>"
}}

CONCEPT: {concept}
COUNTRY: {culture}
CATEGORY: {category}
ANSWER:"""

CLEAN_PROMPT = """You write a compact visual knowledge string for a text-to-image region expert.

Use ONLY the FACTS below (do not invent). Keep culturally specific VISUAL attributes:
material, color/palette, technique, decorations/motifs, and form. Remove redundancy,
duplicated labels, and non-visual metadata (dates, museums, people, collections).
If the facts contain almost no visual detail, output the most distinctive visual traits
that are directly implied by the object name and facts, and nothing else.
One line, <= 45 words, comma/semicolon separated phrases, no full sentences, no quotes.

OBJECT: {concept}  (culture: {culture})
FACTS:
{facts}
ANSWER:"""

FAM_PROMPT = """Rate how commonly a typical person knows this cultural object, from 0 (very obscure,
most people have never heard of it) to 1 (extremely well known worldwide).
Consider world knowledge, not just one country. Return ONLY JSON: {{"familiarity": 0.0}}

OBJECT: {concept}  (culture: {culture})
ANSWER:"""


def _empty_facts(facts) -> bool:
    return not facts or not any(v for v in facts.values())


def parse_concept(concept: str, culture: str, category: str, model: str = "gpt-4o-mini") -> dict:
    txt = chat(PARSE_PROMPT.format(concept=concept, culture=culture, category=category),
               model=model, json_mode=True, max_tokens=220)
    return parse_json(txt)


def _facts_to_text(facts: dict) -> str:
    return "\n".join(f"- {k}: {', '.join(map(str, v))}" for k, v in facts.items() if v)


VERIFY_PROMPT = """You validate a Wikidata candidate for a cultural knowledge base and, if it matches,
write a compact visual knowledge string.

A candidate matches only if it is the SAME object as the CONCEPT (a culturally specific
physical/craft object, dish, textile, vessel, etc.). If the candidate is a place, team,
person, organisation, unrelated thing, or clearly a different object, set match=false.

When match is true, write knowledge_text using ONLY the FACTS (no invention): keep culturally
specific VISUAL attributes (material, color/palette, technique, decorations/motifs, form),
drop metadata (dates, museums, people, collections). One line, <=45 words, comma/semicolon
separated phrases, no quotes.

Return ONLY json:
{{"match": true/false, "reason": "<short>", "knowledge_text": "<...>"}}

CONCEPT: {concept}
EXPECTED CULTURE: {culture}
EXPECTED CATEGORY: {category}
CANDIDATE LABEL: {label}
CANDIDATE DESCRIPTION: {description}
FACTS:
{facts}
ANSWER:"""


def verify_and_clean(concept: str, culture: str, category: str, label: str, description: str,
                     facts: dict, model: str = "gpt-4o-mini") -> dict:
    txt = chat(VERIFY_PROMPT.format(
        concept=concept, culture=culture, category=category, label=label or "",
        description=description or "", facts=_facts_to_text(facts or {})),
        model=model, json_mode=True, max_tokens=220)
    try:
        d = parse_json(txt)
        return {"match": bool(d.get("match")), "reason": d.get("reason", ""),
                "knowledge_text": (d.get("knowledge_text") or "").strip().strip('"')}
    except Exception as exc:  # noqa: BLE001
        return {"match": False, "reason": f"parse_error:{exc}", "knowledge_text": ""}


def clean_knowledge(concept: str, culture: str, facts: dict, model: str = "gpt-4o-mini") -> str:
    txt = chat(CLEAN_PROMPT.format(concept=concept, culture=culture, facts=_facts_to_text(facts)),
               model=model, max_tokens=160, temperature=0.0)
    return txt.strip().strip('"')


def familiarity(concept: str, culture: str, model: str = "gpt-4o-mini") -> float:
    txt = chat(FAM_PROMPT.format(concept=concept, culture=culture),
               model=model, json_mode=True, max_tokens=30)
    try:
        return float(parse_json(txt)["familiarity"])
    except Exception:
        return float("nan")


def llm_fallback(concept: str, culture: str, category: str, model: str = "gpt-4o") -> str:
    """Generate visual knowledge when no external retrieval is available."""
    prompt = (
        "Write a compact visual knowledge string for a text-to-image region expert. "
        "Give culturally specific VISUAL attributes: material, color/palette, technique, "
        "decorations/motifs, form. One line, <=45 words, comma-separated phrases, no quotes.\n"
        f"OBJECT: {concept} (culture: {culture}, category: {category})\nANSWER:"
    )
    return chat(prompt, model=model, max_tokens=160, temperature=0.0).strip().strip('"')


# ---------------- batched variants (many items per request) ----------------

def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _parse_results(txt, expected: int):
    d = parse_json(txt)
    res = d.get("results", d if isinstance(d, list) else [])
    out = {}
    for r in res:
        try:
            out[int(r["i"])] = r
        except Exception:
            continue
    return out


VERIFY_BATCH = """For EACH numbered item, decide whether the Wikidata CANDIDATE is the SAME object as the
CONCEPT (a culturally specific physical/craft object, dish, textile, vessel...). If it is a place,
team, person, organisation, unrelated or different object -> match=false. If match=true, write
knowledge_text using ONLY the facts (no invention): culturally specific VISUAL attributes
(material, color/palette, technique, decorations/motifs, form), <=45 words, comma-separated
phrases, no metadata, no quotes.

Return ONLY json: {{"results":[{{"i":0,"match":true,"knowledge_text":"..."}}, ...]}}

ITEMS:
{items}
ANSWER:"""


def batch_verify_clean(items: list[dict], model: str = "gpt-4o-mini", chunk: int = 20) -> dict:
    """items: [{concept,culture,category,label,description,facts}] -> {index: {match,knowledge_text}}."""
    out = {}
    for grp in _chunks(list(enumerate(items)), chunk):
        lines = []
        for i, it in grp:
            lines.append(
                f"#{i} CONCEPT: {it['concept']} | CULTURE: {it.get('culture','')} | CATEGORY: {it.get('category','')}\n"
                f"   CANDIDATE: {it.get('label','')} :: {it.get('description','')}\n"
                f"   FACTS: {_facts_to_text(it.get('facts') or {}) or '-'}"
            )
        try:
            txt = chat(VERIFY_BATCH.format(items="\n".join(lines)), model=model, json_mode=True, max_tokens=1600)
            out.update(_parse_results(txt, len(grp)))
        except Exception as exc:  # noqa: BLE001
            for i, _ in grp:
                out[i] = {"match": False, "knowledge_text": "", "reason": f"batch_error:{exc}"}
    return out


FAM_BATCH = """Rate how commonly a typical person knows each cultural object from 0 (obscure) to 1
(world-famous). Return ONLY json: {{"results":[{{"i":0,"familiarity":0.0}}, ...]}}

ITEMS:
{items}
ANSWER:"""


def batch_familiarity(items: list[dict], model: str = "gpt-4o-mini", chunk: int = 25) -> dict:
    out = {}
    for grp in _chunks(list(enumerate(items)), chunk):
        lines = [f"#{i} {it['concept']} ({it.get('culture','')})" for i, it in grp]
        try:
            txt = chat(FAM_BATCH.format(items="\n".join(lines)), model=model, json_mode=True, max_tokens=900)
            for i, r in _parse_results(txt, len(grp)).items():
                try:
                    out[i] = float(r["familiarity"])
                except Exception:
                    out[i] = float("nan")
        except Exception:  # noqa: BLE001
            for i, _ in grp:
                out[i] = float("nan")
    return out


FB_BATCH = """Write a compact visual knowledge string for EACH item: culturally specific VISUAL
attributes (material, color/palette, technique, decorations/motifs, form). One line, <=45 words,
comma-separated phrases, no quotes. Return ONLY json: {{"results":[{{"i":0,"knowledge_text":"..."}},...]}}

ITEMS:
{items}
ANSWER:"""


def batch_fallback(items: list[dict], model: str = "gpt-4o", chunk: int = 20) -> dict:
    out = {}
    for grp in _chunks(list(enumerate(items)), chunk):
        lines = [f"#{i} {it['concept']} (culture: {it.get('culture','')}, category: {it.get('category','')})"
                 for i, it in grp]
        try:
            txt = chat(FB_BATCH.format(items="\n".join(lines)), model=model, json_mode=True, max_tokens=1600)
            for i, r in _parse_results(txt, len(grp)).items():
                out[i] = (r.get("knowledge_text") or "").strip().strip('"')
        except Exception:  # noqa: BLE001
            for i, _ in grp:
                out[i] = ""
    return out
