"""MM-DiT causal binding experiment library (SD3.5 Large).

Core claim under test (C1): in cultural compositional binding failures, the causal
locus is the attention-written *content* (value / attn_gated), not the *routing*
(attention weights) nor the gate.

2x2 factorial over an active (layer x step) window:
  baseline  : no patch
  w_fix     : replace target query rows of the attention weights with donor rows
  v_fix     : replace target token values with donor values
  both_fix  : both

Donor = the B-only conditional branch of the SAME forward (batch index 3), with the
B object placed at the same spatial position (right) as in the mixed scene, so the
patch is position-aligned (no resampling).
"""
from __future__ import annotations

import hashlib
import json
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_DIR = "/science/wx/pry/models/stable-diffusion-3.5-large"
SAM_DIR = "/science/wx/pry/models/sam-vit-base"

from pairs import PAIRS, a_only_prompt, b_donor_prompt, b_only_prompt, ss_prompt  # noqa: E402,F401


def make_latent(pipe, seed: int, batch: int, height: int, width: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    latent = pipe.prepare_latents(
        1,
        pipe.transformer.config.in_channels,
        height,
        width,
        torch.float16,
        torch.device("cuda"),
        generator,
        None,
    )
    digest = hashlib.sha256(latent.detach().cpu().numpy().tobytes()).hexdigest()
    return latent.repeat(batch, 1, 1, 1), digest


class Controller:
    """Holds the patch config and per-forward state for an active (layer x step) window."""

    def __init__(self, grid: int, layers, steps, arm: str):
        assert arm in ("baseline", "w_fix", "v_fix", "both_fix")
        self.grid = grid
        self.layers = tuple(int(x) for x in layers)
        self.steps = tuple(int(x) for x in steps)
        self.arm = arm
        self.step_idx = -1
        self.capture: dict = {}
        self.capture_enabled = False

    # ---- step bookkeeping -------------------------------------------------
    def pre(self, module, args, kwargs):
        self.step_idx += 1

    def reset(self):
        self.step_idx = -1

    def active(self, layer: int) -> bool:
        return self.step_idx in self.steps and layer in self.layers

    # ---- spatial region ---------------------------------------------------
    def target_indices(self, device):
        """Right-object image-token indices (row-major on the token grid)."""
        g = self.grid
        y0, y1 = max(1, g // 8), min(g - 1, 7 * g // 8)
        x0, x1 = g // 2, min(g - 1, 15 * g // 16)
        idx = [y * g + x for y in range(y0, y1) for x in range(x0, x1)]
        return torch.tensor(idx, dtype=torch.long, device=device)

    # ---- patches ----------------------------------------------------------
    def patch_weights(self, layer, weights):
        """weights: [B=4, heads, seq_q, seq_k]; replace target query rows of branch 2 with branch 3."""
        if self.arm not in ("w_fix", "both_fix") or not self.active(layer):
            return weights
        idx = self.target_indices(weights.device)
        out = weights.clone()
        out[2].index_copy_(1, idx, out[3].index_select(1, idx))
        return out

    def patch_value(self, layer, value):
        """value: [B=4, heads, seq, head_dim]; replace target token values of branch 2 with branch 3."""
        if self.arm not in ("v_fix", "both_fix") or not self.active(layer):
            return value
        idx = self.target_indices(value.device)
        out = value.clone()
        out[2].index_copy_(1, idx, out[3].index_select(1, idx))
        return out

    # ---- capture (E1 decomposition) --------------------------------------
    def maybe_capture(self, layer, name, tensor):
        if self.capture_enabled and self.active(layer):
            self.capture.setdefault(layer, {})[name] = tensor.detach().float().cpu()


class JointProcessor:
    """Attention processor with optional routing/value patching.

    Uses explicit softmax(qk^T/sqrt(d)) so the routing weights are materialized
    and can be patched.
    """

    def __init__(self, controller: Controller, layer: int):
        self.controller = controller
        self.layer = layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        batch_size = hidden_states.shape[0]
        image_tokens = hidden_states.shape[1]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if encoder_hidden_states is not None:
            cq = attn.add_q_proj(encoder_hidden_states)
            ck = attn.add_k_proj(encoder_hidden_states)
            cv = attn.add_v_proj(encoder_hidden_states)
            cq = cq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ck = ck.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            cv = cv.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                cq = attn.norm_added_q(cq)
            if attn.norm_added_k is not None:
                ck = attn.norm_added_k(ck)
            query, key, value = (
                torch.cat([query, cq], 2),
                torch.cat([key, ck], 2),
                torch.cat([value, cv], 2),
            )
        value = self.controller.patch_value(self.layer, value)
        scale = head_dim ** -0.5
        weights = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) * scale, dim=-1)
        weights = self.controller.patch_weights(self.layer, weights)
        states = torch.matmul(weights, value)
        combined = states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        enc_out = None
        if encoder_hidden_states is not None:
            combined, enc_out = combined[:, :image_tokens], combined[:, image_tokens:]
            if not attn.context_pre_only:
                enc_out = attn.to_add_out(enc_out)
        combined = attn.to_out[1](attn.to_out[0](combined))
        return (combined, enc_out) if encoder_hidden_states is not None else combined


def install(pipe, controller: Controller):
    """Replace the active layers' attention processors and block forwards.

    The block forward mirrors diffusers JointTransformerBlock exactly, while
    exposing the decomposition tensors for capture. Safe to call repeatedly.
    """
    for layer in controller.layers:
        block = pipe.transformer.transformer_blocks[layer]
        if not hasattr(block, "_orig_forward"):
            block._orig_forward = block.forward
            block._orig_processor = block.attn.processor
        block.attn.set_processor(JointProcessor(controller, layer))

        def forward(self, hidden_states, encoder_hidden_states, temb, joint_attention_kwargs=None, _layer=layer):
            joint_attention_kwargs = joint_attention_kwargs or {}
            h_in = hidden_states
            if self.use_dual_attention:
                norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_h2, gate_msa2 = self.norm1(h_in, emb=temb)
            else:
                norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(h_in, emb=temb)
            if self.context_pre_only:
                norm_ctx = self.norm1_context(encoder_hidden_states, temb)
                c_gate_msa = None
            else:
                norm_ctx, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
            attn_raw, context_attn_output = self.attn(
                hidden_states=norm_h, encoder_hidden_states=norm_ctx, **joint_attention_kwargs
            )
            attn_gated = gate_msa.unsqueeze(1) * attn_raw
            if self.use_dual_attention:
                attn_gated = attn_gated + gate_msa2.unsqueeze(1) * self.attn2(
                    hidden_states=norm_h2, **joint_attention_kwargs
                )
            h_attn = h_in + attn_gated
            x_norm = self.norm2(h_attn)
            x_mod = x_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            if self._chunk_size is not None:
                raise RuntimeError("feed-forward chunking must be disabled")
            f_raw = self.ff(x_mod)
            f_gated = gate_mlp.unsqueeze(1) * f_raw
            h_out = h_attn + f_gated
            controller.maybe_capture(_layer, "h_in", h_in)
            controller.maybe_capture(_layer, "gate_msa", gate_msa)
            controller.maybe_capture(_layer, "gate_mlp", gate_mlp)
            controller.maybe_capture(_layer, "attn_raw", attn_raw)
            controller.maybe_capture(_layer, "attn_gated", attn_gated)
            controller.maybe_capture(_layer, "h_attn", h_attn)
            controller.maybe_capture(_layer, "f_raw", f_raw)
            controller.maybe_capture(_layer, "f_gated", f_gated)
            controller.maybe_capture(_layer, "h_out", h_out)
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
                norm_ctx2 = self.norm2_context(encoder_hidden_states)
                norm_ctx2 = norm_ctx2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_ctx2)
            return encoder_hidden_states, h_out

        block.forward = types.MethodType(forward, block)
    old_hook = getattr(pipe.transformer, "_exp_pre_hook", None)
    if old_hook is not None:
        old_hook.remove()
    pipe.transformer._exp_pre_hook = pipe.transformer.register_forward_pre_hook(controller.pre, with_kwargs=True)


def run_arm(pipe, controller: Controller, pair_id: int, seed: int, steps: int, size: int,
            guidance: float, out_dir: Path, arm_name: str):
    """Run one arm: mixed AB (target) + B-only at same position (donor) in one forward."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"pair{pair_id}_seed{seed}_{arm_name}.png"
    donor_path = out_dir / f"pair{pair_id}_seed{seed}_{arm_name}_donorB.png"
    if target_path.exists() and donor_path.exists():
        return {"event": "skipped", "path": str(target_path)}
    latent, latent_sha = make_latent(pipe, seed, 2, size, size)
    controller.reset()
    prompts = [ss_prompt(pair_id), b_donor_prompt(pair_id)]
    t0 = time.time()
    with torch.inference_mode():
        images = pipe(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            negative_prompt=["", ""],
            negative_prompt_2=["", ""],
            negative_prompt_3=["", ""],
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            latents=latent,
            output_type="pil",
        ).images
    images[0].save(target_path)
    images[1].save(donor_path)
    meta = {
        "pair": pair_id,
        "seed": seed,
        "arm": arm_name,
        "layers": list(controller.layers),
        "steps_active": list(controller.steps),
        "steps": steps,
        "size": size,
        "guidance": guidance,
        "latent_sha256": latent_sha,
        "ss_prompt": prompts[0],
        "donor_prompt": prompts[1],
        "seconds": round(time.time() - t0, 2),
        "peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    }
    (out_dir / f"pair{pair_id}_seed{seed}_{arm_name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"event": "completed", **meta, "path": str(target_path)}
