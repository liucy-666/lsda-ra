from __future__ import annotations

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)


def prepare_schedule(pipe, latents, steps: int):
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None):
        _, _, height, width = latents.shape
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        scheduler_kwargs["mu"] = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, steps, pipe._execution_device, **scheduler_kwargs
    )
    return timesteps, scheduler_kwargs.get("mu")


def transformer_pair(pipe, latents, timestep, prompt_embeds, pooled_prompt_embeds):
    latent_input = torch.cat([latents, latents], dim=0)
    return pipe.transformer(
        hidden_states=latent_input,
        timestep=timestep.expand(2),
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]


def region_cfg_score(
    pipe,
    crop,
    timestep,
    negative,
    positive,
    negative_pooled,
    positive_pooled,
    scale: float,
):
    pair_embeds = torch.cat([negative, positive], dim=0)
    pair_pooled = torch.cat([negative_pooled, positive_pooled], dim=0)
    uncond, cond = transformer_pair(
        pipe, crop, timestep, pair_embeds, pair_pooled
    ).chunk(2)
    return uncond + scale * (cond - uncond)


def encode_prompts(pipe, prompts, max_sequence_length: int = 256):
    positives = []
    positive_pooled = []
    negative_embeds = None
    negative_pooled = None
    for index, prompt in enumerate(prompts):
        prompt_embed, negative_embed, pooled, negative_pool = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            do_classifier_free_guidance=True,
            device=pipe._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        positives.append(prompt_embed)
        positive_pooled.append(pooled)
        if index == 0:
            negative_embeds = negative_embed
            negative_pooled = negative_pool
    return {
        "positive": torch.cat(positives, dim=0),
        "negative": negative_embeds,
        "positive_pooled": torch.cat(positive_pooled, dim=0),
        "negative_pooled": negative_pooled,
    }
