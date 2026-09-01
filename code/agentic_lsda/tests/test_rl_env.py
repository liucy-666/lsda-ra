from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_env import (
    ActionKind,
    AgentAction,
    AgenticLsdaEnv,
    ConstrainedReward,
    EnvConfig,
    Observation,
    QualityScore,
    RewardConfig,
)


class FakeBackend:
    def __init__(self):
        self.step_index = 0
        self.binding = 0.2
        self.structure = 0.9
        self.snapshots = {}

    def observation(self):
        return Observation(
            sample_id="demo",
            step=self.step_index,
            num_steps=28,
            preview_path=None,
            entity_metrics={},
            global_metrics={},
            quality=QualityScore(self.binding, self.structure, 0.4, 0.1),
        )

    def reset(self, task):
        self.__init__()
        return self.observation()

    def advance(self, steps):
        self.step_index += steps
        return self.observation()

    def snapshot(self):
        checkpoint = f"c{len(self.snapshots)}"
        self.snapshots[checkpoint] = (self.step_index, self.binding, self.structure)
        return checkpoint

    def invoke_lsda(self, target, rollback_steps, observe_after_steps):
        self.step_index = max(0, self.step_index - rollback_steps) + observe_after_steps
        self.binding += 0.5
        self.structure -= 0.1
        return self.observation()

    def accept(self, checkpoint_id):
        return self.observation()

    def rollback(self, checkpoint_id):
        self.step_index, self.binding, self.structure = self.snapshots[checkpoint_id]
        return self.observation()

    def is_terminal(self):
        return self.step_index >= 28


class RlEnvironmentTest(unittest.TestCase):
    def make_env(self):
        reward = ConstrainedReward(
            RewardConfig(
                binding_weight=1.0,
                structure_weight=1.0,
                leakage_weight=0.0,
                artifact_weight=0.0,
                cost_weight=0.1,
                structure_floor=0.7,
                constraint_penalty=2.0,
            )
        )
        return AgenticLsdaEnv(
            FakeBackend(),
            EnvConfig(
                decision_stride=4,
                post_invoke_observation_steps=2,
                tool_budget=2,
                invoke_cost=1.0,
            ),
            reward,
        )

    def test_repeated_wait_invoke_feedback_accept(self):
        env = self.make_env()
        observation, info = env.reset({"sample_id": "demo"})
        self.assertEqual(observation.step, 0)
        observation = env.step(AgentAction(ActionKind.WAIT)).observation
        self.assertEqual(observation.step, 4)
        invoked = env.step(AgentAction(ActionKind.INVOKE, target="B"))
        self.assertEqual(invoked.observation.step, 6)
        self.assertGreater(invoked.reward.total, 0)
        self.assertTrue(invoked.info["pending_intervention"])
        accepted = env.step(AgentAction(ActionKind.ACCEPT))
        self.assertFalse(accepted.info["pending_intervention"])
        reinvoked = env.step(AgentAction(ActionKind.INVOKE, target="A", rollback_steps=2))
        self.assertEqual(reinvoked.info["tool_budget_remaining"], 0)

    def test_rollback_restores_preinvoke_state(self):
        env = self.make_env()
        env.reset({})
        env.step(AgentAction(ActionKind.WAIT))
        env.step(AgentAction(ActionKind.INVOKE, target="AB"))
        rolled_back = env.step(AgentAction(ActionKind.ROLLBACK))
        self.assertEqual(rolled_back.observation.step, 4)
        self.assertAlmostEqual(rolled_back.observation.quality.binding, 0.2)

    def test_pending_intervention_blocks_wait(self):
        env = self.make_env()
        env.reset({})
        env.step(AgentAction(ActionKind.INVOKE, target="A"))
        with self.assertRaises(RuntimeError):
            env.step(AgentAction(ActionKind.WAIT))


if __name__ == "__main__":
    unittest.main()
