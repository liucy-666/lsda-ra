from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from timed_lsda import _selected_indices, denoise_timed_specialists


class FakeScheduler:
    def step(self, score, timestep, sample, return_dict=False):
        return (sample + 1.0,)


class FakeTransformerConfig:
    patch_size = 1


class FakeTransformer:
    config = FakeTransformerConfig()


class FakePipe:
    scheduler = FakeScheduler()
    transformer = FakeTransformer()


def prepare_schedule(pipe, latent, steps):
    return torch.arange(steps, 0, -1, dtype=torch.float32), 0.5


def region_cfg_score(*args, **kwargs):
    crop = args[1]
    return torch.zeros_like(crop)


def owner_bounds(owner, patch_size):
    ys, xs = torch.where(owner[0, 0] > 0.5)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class TimedLsdaTest(unittest.TestCase):
    def test_target_mapping(self):
        self.assertEqual(_selected_indices("A", 2), {0})
        self.assertEqual(_selected_indices("B", 2), {1})
        self.assertEqual(_selected_indices("AB", 2), {0, 1})

    def test_late_branch_uses_previous_native_then_preserves_unselected_owner(self):
        initial = torch.zeros((1, 1, 2, 2))
        native_states = [
            torch.full_like(initial, 10.0),
            torch.full_like(initial, 20.0),
            torch.full_like(initial, 30.0),
        ]
        owner_a = torch.zeros_like(initial)
        owner_b = torch.zeros_like(initial)
        owner_a[:, :, :, 0] = 1.0
        owner_b[:, :, :, 1] = 1.0
        background = torch.zeros_like(initial)
        encoded = {
            "negative": torch.zeros((1, 1)),
            "positive": torch.zeros((3, 1)),
            "negative_pooled": torch.zeros((1, 1)),
            "positive_pooled": torch.zeros((3, 1)),
        }

        final, audit = denoise_timed_specialists(
            pipe=FakePipe(),
            initial=initial,
            encoded=encoded,
            owners=[owner_a, owner_b],
            background=background,
            native_states=native_states,
            helper_functions=(prepare_schedule, None, region_cfg_score, owner_bounds),
            intervention_step=1,
            target="A",
            guidance_scale=4.5,
        )

        self.assertTrue(torch.equal(final[:, :, :, 0], torch.full((1, 1, 2), 12.0)))
        self.assertTrue(torch.equal(final[:, :, :, 1], torch.full((1, 1, 2), 30.0)))
        self.assertEqual(audit["intervention_step"], 1)
        self.assertEqual(audit["selected_entity_indices"], [0])
        self.assertEqual([row["step"] for row in audit["step_audit"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
