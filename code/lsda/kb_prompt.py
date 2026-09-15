"""Inject audited KB text into LSDA prompt configs without changing attr text.

The clean v1 runtime still receives one prompt per expert. This adapter keeps the
original ``attr`` verbatim for comparability and appends the optional, audited
``knowledge_text`` as a clearly marked visual hint.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping


def _article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def knowledge_fragment(entity: Mapping[str, Any]) -> str:
    value = str(entity.get("knowledge_text") or "").strip()
    return value.rstrip(" ;,.\n")


def entity_prompt(entity: Mapping[str, Any], side: str, *, include_knowledge: bool = True) -> str:
    existing = str(entity.get("prompt") or "").strip()
    if existing:
        fragment = knowledge_fragment(entity)
        if include_knowledge and fragment and fragment.lower() not in existing.lower():
            return existing.rstrip(" .") + f"; verified visual knowledge: {fragment}."
        return existing
    name = str(entity.get("name") or entity.get("concept") or "").strip()
    attr = str(entity.get("attr") or entity.get("concept") or name).strip()
    prefix = "" if name.lower().startswith(("a ", "an ", "the ")) else _article(name) + " "
    prompt = f"{prefix}{name} on the {side}, {attr}"
    if include_knowledge and knowledge_fragment(entity):
        prompt += f"; verified visual knowledge: {knowledge_fragment(entity)}"
    return prompt + ", photorealistic, sharp focus, neutral studio background."


def inject_knowledge_text(config: Mapping[str, Any], *, include_knowledge: bool = True) -> dict[str, Any]:
    """Return a copy of a clean-v1 config with knowledge-aware entity prompts."""
    output = copy.deepcopy(dict(config))
    entities = output.get("entities") or []
    for index, entity in enumerate(entities):
        side = str(entity.get("position") or ("left" if index == 0 else "right"))
        entity["prompt"] = entity_prompt(entity, side, include_knowledge=include_knowledge)
    return output


def inject_from_kb(
    config: Mapping[str, Any], rows: Iterable[Mapping[str, Any]], *, include_knowledge: bool = True
) -> dict[str, Any]:
    """Join KB rows onto config entities and then inject their text.

    A config entity should carry ``kb_key`` (preferred), ``concept``, or
    ``culture_noun``. Existing prompts are preserved and only receive the
    verified knowledge suffix.
    """
    output = copy.deepcopy(dict(config))
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for field in ("key", "concept", "culture_noun", "name"):
            value = str(row.get(field) or "").strip().lower()
            if value:
                index[value] = row
    for entity in output.get("entities") or []:
        key = next((str(entity.get(field) or "").strip().lower()
                    for field in ("kb_key", "key", "concept", "culture_noun", "name")
                    if str(entity.get(field) or "").strip().lower()), "")
        row = index.get(key)
        if row and knowledge_fragment(row):
            entity["knowledge_text"] = knowledge_fragment(row)
    return inject_knowledge_text(output, include_knowledge=include_knowledge)


__all__ = ["entity_prompt", "inject_from_kb", "inject_knowledge_text", "knowledge_fragment"]
