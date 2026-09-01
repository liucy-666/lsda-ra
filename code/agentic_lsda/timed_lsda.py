from __future__ import annotations

import copy
from typing import Iterable

import torch


def _selected_indices(target: str, entity_count: int) -> set[int]:
    if target == "AB":
        return set(range(entity_count))
    mapping = {"A": 0, "B": 1}
    if target not in mapping or mapping[target] >= entity_count:
        raise ValueError(f"target {target!r} is invalid for {entity_count} entities")
    return {mapping[target]}


def denoise_timed_specialists(
    pipe,
    initial: torch.Tensor,
    encoded: dict,
    owners: list[torch.Tensor],
    background: torch.Tensor,
    native_states: list[torch.Tensor],
    helper_functions: tuple,
    intervention_step: int,
    target: str,
    guidance_scale: float,
):
    """Branch from a native checkpoint and apply LSDA to selected owners.

    `native_states[k]` is the state after scheduler step k.  Therefore a branch at
    step k reads `initial` when k == 0 and `native_states[k - 1]` otherwise.
    Non-selected owners and the background remain on the frozen native trajectory.
    """
    prepare_schedule, _, region_cfg_score, owner_bounds = helper_functions
    if not 0 <= intervention_step < len(native_states):
        raise ValueError(
            f"intervention_step must be in [0, {len(native_states) - 1}], got {intervention_step}"
        )
    selected = _selected_indices(target, len(owners))
    latent = initial.clone() if intervention_step == 0 else native_states[intervention_step - 1].clone()
    timesteps, mu = prepare_schedule(pipe, initial, len(native_states))
    patch_size = int(pipe.transformer.config.patch_size)
    bounds = [owner_bounds(owner, patch_size) for owner in owners]
    schedulers = {index: copy.deepcopy(pipe.scheduler) for index in selected}
    audit = []

    with torch.inference_mode():
        for step_index in range(intervention_step, len(timesteps)):
            timestep = timesteps[step_index]
            native_next = native_states[step_index]
            next_latent = background * native_next
            committed_deltas = []
            for entity_index, (owner, bounds_i) in enumerate(zip(owners, bounds)):
                if entity_index not in selected:
                    next_latent += owner * native_next
                    continue
                left, top, right, bottom = bounds_i
                crop = latent[:, :, top:bottom, left:right]
                score = region_cfg_score(
                    pipe,
                    crop,
                    timestep,
                    encoded["negative"],
                    encoded["positive"][entity_index + 1 : entity_index + 2],
                    encoded["negative_pooled"],
                    encoded["positive_pooled"][entity_index + 1 : entity_index + 2],
                    guidance_scale,
                )
                candidate = schedulers[entity_index].step(
                    score.to(crop.dtype), timestep, crop, return_dict=False
                )[0].to(initial.dtype)
                owner_crop = owner[:, :, top:bottom, left:right]
                next_latent[:, :, top:bottom, left:right] += owner_crop * candidate
                delta = torch.zeros_like(latent)
                delta[:, :, top:bottom, left:right] = owner_crop * (candidate - crop)
                committed_deltas.append((entity_index, delta))

            latent = next_latent.to(initial.dtype)
            audit.append(
                {
                    "step": step_index,
                    "timestep": float(timestep.item()),
                    "target": target,
                    "state_rms": float(latent.square().mean().sqrt().item()),
                    "native_delta_rms": float(
                        (latent - native_next).square().mean().sqrt().item()
                    ),
                    "outside_write_rms": {
                        str(index): float(((1.0 - owners[index]) * delta).square().mean().sqrt().item())
                        for index, delta in committed_deltas
                    },
                    "background_native_match_rms": float(
                        (background * (latent - native_next)).square().mean().sqrt().item()
                    ),
                }
            )
    return latent, {
        "mu": mu,
        "intervention_step": intervention_step,
        "target": target,
        "selected_entity_indices": sorted(selected),
        "bounds": bounds,
        "step_audit": audit,
    }
