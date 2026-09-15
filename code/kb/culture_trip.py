"""Culture-TRIP iterative prompt refinement (Jeong et al., NAACL 2025).

Adapted to use the local Wikidata KB as the external retrieval source and the openlux
LLM as the refiner/scorer. The refinement loop mirrors the paper:
  retrieve -> refine -> score (5 criteria) -> feedback -> repeat
until Total_score >= threshold or max_iters, then return the refined prompt.
"""
from __future__ import annotations

try:
    from .llm_client import chat, parse_json
except ImportError:
    from llm_client import chat, parse_json

import re
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

REFINE_TEMPLATE = """### Instruction:
Please refine the BASE PROMPT by referring to the INFORMATION and FEEDBACK.
The REFINED PROMPT should be written so that someone unfamiliar with the CULTURE NOUN can read it and draw a picture of it.

Since the INFORMATION may contain incorrect details, carefully verify that it pertains to the CULTURE NOUN before using it.

If the BASE PROMPT cannot adequately accommodate the CULTURE NOUN, slight modifications are allowed. When adding additional information to a single sentence to provide sufficient detail, expand the original sentence into three sentences.
However, the scene depicted in the BASE PROMPT must remain unchanged.

### TODO:
CULTURE NOUN:
{culture_noun}

INFORMATION:
{information}

BASE PROMPT:
{prompt}

BEFORE REFINED PROMPT:
{refined_prompt}

REFINE FEEDBACK:
{feedback}

ANSWER:
"""

SCORING_TEMPLATE = """### Instruction:
Please evaluate the REFINED PROMPT with 5 criteria (Clarity, Visual_detail, Background, Purpose, Comparable_object).
- Clarity: How clear and easy to understand the prompt is, and whether it uses only the information necessary to describe the CULTURE NOUN.
- Visual_detail: Whether the prompt provides a sufficient amount of visual information, such as colors, shapes, etc.
- Background: Whether the historical or temporal background information provided in the prompt is appropriate.
- Purpose: Whether the description of the intended use or the users of the subject in the prompt is appropriate.
- Comparable_object: How well the prompt compares to existing well-known or popular examples.
Each criterion cannot exceed a score of 10. Provide each criterion's score and the total score.
Write only the score, not the description.

ANSWER FORMAT:
{{
    "Clarity" : 5,
    "Visual_detail" : 5,
    "Background" : 5,
    "Purpose" : 5,
    "Comparable_object" : 5,
    "Total_score" : 25
}}

### TODO:
CULTURE NOUN:
{culture_noun}

REFINED PROMPT:
{refined_prompt}

ANSWER:
"""

FEEDBACK_TEMPLATE = """### Instruction:
Review the items of SCORE and provide feedback on how to improve each item's score, specifically focusing on the modification of REFINED PROMPT about CULTURE NOUN.

### TODO:
CULTURE NOUN:
{culture_noun}

REFINED PROMPT:
{refined_prompt}

SCORE:
{score}

ANSWER:
"""


def format_information(kb_entry: dict) -> str:
    """Render a Wikidata KB entry as the 'INFORMATION' block."""
    wd = (kb_entry or {}).get("wikidata") or {}
    facts = wd.get("facts") or {}
    lines = []
    label = wd.get("label") or kb_entry.get("culture_noun", "")
    desc = wd.get("description", "")
    if label:
        lines.append(f"Name: {label}")
    if desc:
        lines.append(f"Summary: {desc}")
    pretty = {
        "instance_of": "Type", "subclass_of": "Subclass of", "country_of_origin": "Country of origin",
        "country": "Country", "material_used": "Material", "depicts": "Depicts", "genre": "Genre",
        "location": "Location", "part_of": "Part of", "has_use": "Use", "culture": "Culture",
    }
    for key, name in pretty.items():
        if facts.get(key):
            lines.append(f"{name}: {', '.join(map(str, facts[key][:6]))}")
    return "\n".join(lines) if lines else "(no structured knowledge retrieved)"


def refine(culture_noun, information, base_prompt, refined_prompt, feedback, model):
    return chat(REFINE_TEMPLATE.format(
        culture_noun=culture_noun, information=information, prompt=base_prompt,
        refined_prompt=refined_prompt, feedback=feedback), model=model, max_tokens=900).strip()


def score(culture_noun, refined_prompt, model):
    txt = chat(SCORING_TEMPLATE.format(culture_noun=culture_noun, refined_prompt=refined_prompt),
               model=model, json_mode=True, max_tokens=300)
    return parse_json(txt)


def feedback(culture_noun, refined_prompt, score_obj, model):
    return chat(FEEDBACK_TEMPLATE.format(
        culture_noun=culture_noun, refined_prompt=refined_prompt,
        score=score_obj), model=model, max_tokens=500).strip()


def culture_trip(culture_noun: str, base_prompt: str, information: str,
                 threshold: int = 40, max_iters: int = 5, model: str = "gpt-5.4",
                 verbose: bool = False) -> dict:
    refined, fb = "", ""
    history = []
    for it in range(max_iters):
        refined = refine(culture_noun, information, base_prompt, refined, fb, model)
        sc = score(culture_noun, refined, model)
        history.append({"iter": it, "score": sc, "refined_prompt": refined})
        if verbose:
            print(f"[iter {it}] total={sc.get('Total_score')}")
        if sc.get("Total_score", 0) >= threshold:
            break
        fb = feedback(culture_noun, refined, sc, model)
    return {"refined_prompt": refined, "final_score": history[-1]["score"] if history else None,
            "iterations": len(history), "history": history}


# Deterministic fallback used by the batch KB builder. The LLM loop above is
# optional enrichment; these helpers keep facts and provenance usable offline.
def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" ;,.")


def facts_to_text(facts: Union[Mapping[str, Any], Iterable[Tuple[str, Any]]]) -> str:
    items = facts.items() if isinstance(facts, Mapping) else facts
    parts = []
    for key, value in sorted(items, key=lambda pair: str(pair[0])):
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(sorted(dict.fromkeys(_clean(item) for item in value if _clean(item))))
        elif isinstance(value, Mapping):
            value = ", ".join(f"{k}: {_clean(v)}" for k, v in sorted(value.items()) if v not in (None, "", [], {}))
        value = _clean(value)
        if value:
            parts.append(f"{str(key).replace('_', ' ')}: {value}")
    return "; ".join(parts)


def build_knowledge_text(
    facts: Mapping[str, Any], *, llm: Optional[Any] = None, max_chars: int = 500
) -> Tuple[str, str]:
    """Return ``(knowledge_text, origin)`` with a deterministic safe fallback."""
    fallback = facts_to_text(facts)
    if llm is None or not fallback:
        return fallback, "wikidata" if fallback else "empty"
    try:
        result = llm.complete_json(
            "Use only supplied facts. Return JSON with one key knowledge_text; "
            "keep it factual, visual, concise, and do not invent details.",
            "Facts (do not add any):\n" + fallback,
        )
        candidate = result.get("knowledge_text") if isinstance(result, Mapping) else result
        text = _clean(candidate)
        if text and len(text) <= max_chars:
            return text, "llm_refined"
    except Exception:
        pass
    return fallback[:max_chars].rstrip(" ;,.") , "wikidata"
