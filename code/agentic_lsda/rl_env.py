from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ActionKind(str, Enum):
    WAIT = "wait"
    INVOKE = "invoke_lsda"
    ACCEPT = "accept"
    ROLLBACK = "rollback"
    TERMINATE = "terminate"


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    target: str = "none"
    rollback_steps: int = 0

    def validate(self) -> None:
        if self.kind == ActionKind.INVOKE:
            if self.target not in {"A", "B", "AB"}:
                raise ValueError("INVOKE target must be A, B, or AB")
            if self.rollback_steps < 0:
                raise ValueError("rollback_steps must be non-negative")
        elif self.target != "none" or self.rollback_steps != 0:
            raise ValueError(f"{self.kind.value} cannot carry a target or rollback_steps")


@dataclass(frozen=True)
class QualityScore:
    binding: float
    structure: float
    leakage: float
    artifact: float


@dataclass(frozen=True)
class Observation:
    sample_id: str
    step: int
    num_steps: int
    preview_path: str | None
    entity_metrics: dict[str, dict[str, float]]
    global_metrics: dict[str, float]
    quality: QualityScore | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardConfig:
    binding_weight: float
    structure_weight: float
    leakage_weight: float
    artifact_weight: float
    cost_weight: float
    structure_floor: float
    constraint_penalty: float


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    binding_gain: float
    structure_gain: float
    leakage_reduction: float
    artifact_reduction: float
    action_cost: float
    structure_constraint_violated: bool


class ConstrainedReward:
    """C1 improvement with an explicit C2 floor and an auditable decomposition."""

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(
        self, before: QualityScore, after: QualityScore, action_cost: float
    ) -> RewardBreakdown:
        binding_gain = after.binding - before.binding
        structure_gain = after.structure - before.structure
        leakage_reduction = before.leakage - after.leakage
        artifact_reduction = before.artifact - after.artifact
        violated = after.structure < self.config.structure_floor
        total = (
            self.config.binding_weight * binding_gain
            + self.config.structure_weight * structure_gain
            + self.config.leakage_weight * leakage_reduction
            + self.config.artifact_weight * artifact_reduction
            - self.config.cost_weight * action_cost
        )
        if violated:
            total -= self.config.constraint_penalty
        return RewardBreakdown(
            total=total,
            binding_gain=binding_gain,
            structure_gain=structure_gain,
            leakage_reduction=leakage_reduction,
            artifact_reduction=artifact_reduction,
            action_cost=action_cost,
            structure_constraint_violated=violated,
        )


class DiffusionBackend(Protocol):
    """Minimal backend contract; the policy never receives latent tensors directly."""

    def reset(self, task: dict[str, Any]) -> Observation: ...

    def advance(self, steps: int) -> Observation: ...

    def snapshot(self) -> str: ...

    def invoke_lsda(
        self, target: str, rollback_steps: int, observe_after_steps: int
    ) -> Observation: ...

    def accept(self, checkpoint_id: str) -> Observation: ...

    def rollback(self, checkpoint_id: str) -> Observation: ...

    def is_terminal(self) -> bool: ...


@dataclass(frozen=True)
class EnvConfig:
    decision_stride: int
    post_invoke_observation_steps: int
    tool_budget: int
    invoke_cost: float

    def validate(self) -> None:
        if self.decision_stride <= 0 or self.post_invoke_observation_steps <= 0:
            raise ValueError("decision strides must be positive")
        if self.tool_budget < 0 or self.invoke_cost < 0:
            raise ValueError("tool budget and cost must be non-negative")


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: RewardBreakdown | None
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class AgenticLsdaEnv:
    """State machine for repeated WAIT/INVOKE/feedback/ACCEPT-or-ROLLBACK decisions."""

    def __init__(
        self,
        backend: DiffusionBackend,
        config: EnvConfig,
        reward: ConstrainedReward | None = None,
    ):
        config.validate()
        self.backend = backend
        self.config = config
        self.reward_fn = reward
        self.observation: Observation | None = None
        self.pending_checkpoint: str | None = None
        self.tool_budget_remaining = config.tool_budget
        self.terminated = False

    def reset(self, task: dict[str, Any]) -> tuple[Observation, dict[str, Any]]:
        self.pending_checkpoint = None
        self.tool_budget_remaining = self.config.tool_budget
        self.terminated = False
        self.observation = self.backend.reset(task)
        return self.observation, self._info()

    def _info(self) -> dict[str, Any]:
        return {
            "tool_budget_remaining": self.tool_budget_remaining,
            "pending_intervention": self.pending_checkpoint is not None,
        }

    def _reward(
        self, before: Observation, after: Observation, action_cost: float
    ) -> RewardBreakdown | None:
        if self.reward_fn is None:
            return None
        if before.quality is None or after.quality is None:
            raise RuntimeError("training reward requires calibrated quality scores")
        return self.reward_fn(before.quality, after.quality, action_cost)

    def step(self, action: AgentAction) -> StepResult:
        if self.observation is None:
            raise RuntimeError("reset must be called before step")
        if self.terminated:
            raise RuntimeError("episode is already terminated")
        action.validate()
        before = self.observation
        action_cost = 0.0

        if action.kind == ActionKind.WAIT:
            if self.pending_checkpoint is not None:
                raise RuntimeError("pending LSDA result must be ACCEPTed or ROLLed BACK before WAIT")
            after = self.backend.advance(self.config.decision_stride)
        elif action.kind == ActionKind.INVOKE:
            if self.pending_checkpoint is not None:
                raise RuntimeError("cannot INVOKE while another intervention is pending")
            if self.tool_budget_remaining <= 0:
                raise RuntimeError("LSDA tool budget exhausted")
            self.pending_checkpoint = self.backend.snapshot()
            self.tool_budget_remaining -= 1
            action_cost = self.config.invoke_cost
            after = self.backend.invoke_lsda(
                action.target,
                action.rollback_steps,
                self.config.post_invoke_observation_steps,
            )
        elif action.kind == ActionKind.ACCEPT:
            if self.pending_checkpoint is None:
                raise RuntimeError("ACCEPT requires a pending intervention")
            after = self.backend.accept(self.pending_checkpoint)
            self.pending_checkpoint = None
        elif action.kind == ActionKind.ROLLBACK:
            if self.pending_checkpoint is None:
                raise RuntimeError("ROLLBACK requires a pending intervention")
            after = self.backend.rollback(self.pending_checkpoint)
            self.pending_checkpoint = None
        elif action.kind == ActionKind.TERMINATE:
            if self.pending_checkpoint is not None:
                raise RuntimeError("resolve the pending intervention before TERMINATE")
            after = before
            self.terminated = True
        else:  # pragma: no cover
            raise ValueError(action.kind)

        self.observation = after
        self.terminated = self.terminated or self.backend.is_terminal()
        reward = self._reward(before, after, action_cost)
        return StepResult(
            observation=after,
            reward=reward,
            terminated=self.terminated,
            truncated=False,
            info=self._info(),
        )
