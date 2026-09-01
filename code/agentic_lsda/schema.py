from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SCHEMA_VERSION = "agentic_lsda_replay_v1"
VALID_TARGETS = {"none", "A", "B", "AB"}
VALID_SPLITS = {"train", "validation", "test", "pilot"}


@dataclass(frozen=True)
class ReplayAction:
    kind: Literal["wait", "invoke_lsda"]
    intervention_step: int | None
    target: Literal["none", "A", "B", "AB"]

    def validate(self, num_steps: int) -> None:
        if self.target not in VALID_TARGETS:
            raise ValueError(f"invalid target: {self.target}")
        if self.kind == "wait":
            if self.intervention_step is not None or self.target != "none":
                raise ValueError("wait must use intervention_step=null and target=none")
            return
        if self.kind != "invoke_lsda":
            raise ValueError(f"invalid action kind: {self.kind}")
        if self.intervention_step is None or not 0 <= self.intervention_step < num_steps:
            raise ValueError(
                f"intervention_step must be in [0, {num_steps - 1}], got {self.intervention_step}"
            )
        if self.target == "none":
            raise ValueError("invoke_lsda requires target A, B, or AB")


@dataclass(frozen=True)
class ReplayTask:
    replay_id: str
    sample_id: str
    pair_id: str
    pair_index: int
    seed_group: int
    replicate: int
    latent_seed: int
    global_prompt: str
    entity_A_prompt: str
    entity_B_prompt: str
    native_task_id: str
    source_lsda_task_id: str
    source_transition: str
    cohort: str
    split: str
    action: ReplayAction
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, num_steps: int) -> None:
        if not self.replay_id or not self.sample_id or not self.pair_id:
            raise ValueError("replay_id, sample_id, and pair_id are required")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"invalid split: {self.split}")
        if not self.global_prompt or not self.entity_A_prompt or not self.entity_B_prompt:
            raise ValueError(f"empty prompt in {self.replay_id}")
        self.action.validate(num_steps)


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected schema_version={SCHEMA_VERSION}, got {payload.get('schema_version')}"
        )
    num_steps = int(payload["generation"]["num_inference_steps"])
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest tasks must be a non-empty list")
    seen: set[str] = set()
    for raw in tasks:
        task = ReplayTask(
            **{
                **raw,
                "action": ReplayAction(**raw["action"]),
            }
        )
        task.validate(num_steps)
        if task.replay_id in seen:
            raise ValueError(f"duplicate replay_id: {task.replay_id}")
        seen.add(task.replay_id)


def write_manifest(payload: dict[str, Any], path: Path) -> None:
    validate_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_to_dict(task: ReplayTask) -> dict[str, Any]:
    return asdict(task)
