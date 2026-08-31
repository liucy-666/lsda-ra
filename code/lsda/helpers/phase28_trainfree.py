from __future__ import annotations

import argparse
import hashlib
import json
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL.PngImagePlugin import PngInfo

from phase1_common import MODEL_DIR


ROOT = Path("/root/private_data/Pry/Experience")
OUT = ROOT / "phase28_trainfree"
AB_PROMPT = "A Chinese blue-and-white porcelain plate is next to an Italian maiolica vase."
B_PROMPT = (
    "A single Italian Renaissance maiolica vase with a milky white tin-glazed ceramic surface, "
    "decorated with cobalt, ochre, copper-green, and manganese painted motifs."
)
FORMAL_STEPS = (24, 25, 26, 27)
FORMAL_LAYERS = (33, 34, 35, 36)
HEAD_ORDER = (27, 13, 19, 25, 30, 16, 28, 37, 0, 10)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--names", nargs="+", default=None)
    return parser.parse_args()


def catalog():
    common = {"steps": list(FORMAL_STEPS), "layers": list(FORMAL_LAYERS)}
    rows = [
        {"name": "baseline_AB", "mode": "none", **common},
        {
            "name": "selective_residual_top10_a050",
            "mode": "selective_residual",
            "fraction": 0.10,
            "alpha": 0.50,
            **common,
        },
        {"name": "reference_anchor_a010", "mode": "reference_anchor", "alpha": 0.10, **common},
        {"name": "reference_anchor_a025", "mode": "reference_anchor", "alpha": 0.25, **common},
        {"name": "reference_anchor_a050", "mode": "reference_anchor", "alpha": 0.50, **common},
    ]
    for k in (1, 3, 5, 10, 38):
        heads = list(HEAD_ORDER[:k]) if k < 38 else list(range(38))
        rows.append(
            {
                "name": f"head_top{k:02d}_a100" if k < 38 else "head_all38_a100",
                "mode": "head_patch",
                "heads": heads,
                "alpha": 1.0,
                **common,
            }
        )
    rows.extend(
        [
            {"name": "value_path_a025", "mode": "v_patch", "alpha": 0.25, **common},
            {
                "name": "oproj_top10_a050",
                "mode": "o_patch",
                "fraction": 0.10,
                "alpha": 0.50,
                **common,
            },
            {"name": "mlp_gate_b075", "mode": "mlp_gate", "beta": 0.75, **common},
            {
                "name": "anchor_a025_gate_b075",
                "mode": "anchor_gate",
                "alpha": 0.25,
                "beta": 0.75,
                **common,
            },
        ]
    )
    return rows


def make_latent(pipe, seed, batch, height, width):
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
    def __init__(self, steps, layers, height, width):
        self.active_steps = tuple(steps)
        self.active_layers = tuple(layers)
        self.height = height
        self.width = width
        self.grid_h = height // 16
        self.grid_w = width // 16
        self.step = -1
        self.config = None

    def transformer_pre(self, module, inputs, kwargs):
        self.step += 1

    def reset(self, config):
        self.step = -1
        self.config = config

    def active(self, layer):
        return self.step in self.config["steps"] and layer in self.config["layers"]

    def rect(self, kind):
        gh, gw = self.grid_h, self.grid_w
        y0, y1 = max(1, gh // 8), min(gh - 1, 7 * gh // 8)
        if kind == "target":
            x0, x1 = gw // 2, min(gw - 1, 15 * gw // 16)
        elif kind == "donor":
            x0, x1 = gw // 4, 3 * gw // 4
        else:
            raise ValueError(kind)
        return y0, y1, x0, x1

    def indices(self, kind, device):
        y0, y1, x0, x1 = self.rect(kind)
        return torch.tensor(
            [y * self.grid_w + x for y in range(y0, y1) for x in range(x0, x1)],
            dtype=torch.long,
            device=device,
        )

    def feather(self, device, dtype):
        y0, y1, x0, x1 = self.rect("target")
        height, width = y1 - y0, x1 - x0
        ramp_y, ramp_x = max(2, height // 8), max(2, width // 6)
        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        wy = torch.minimum(torch.minimum((y + 1) / ramp_y, (height - y) / ramp_y), torch.ones_like(y))
        wx = torch.minimum(torch.minimum((x + 1) / ramp_x, (width - x) / ramp_x), torch.ones_like(x))
        weight = (wy[:, None] * wx[None, :]).reshape(-1, 1)
        return weight.to(dtype)

    def aligned_donor(self, donor_tokens):
        # donor_tokens: [..., image_tokens, channels]
        y0, y1, x0, x1 = self.rect("donor")
        ty0, ty1, tx0, tx1 = self.rect("target")
        prefix = donor_tokens.shape[:-2]
        channels = donor_tokens.shape[-1]
        spatial = donor_tokens.reshape(*prefix, self.grid_h, self.grid_w, channels)
        crop = spatial[..., y0:y1, x0:x1, :]
        flat_prefix = 1
        for size in prefix:
            flat_prefix *= size
        crop = crop.reshape(flat_prefix, y1 - y0, x1 - x0, channels).permute(0, 3, 1, 2)
        crop = F.interpolate(crop.float(), size=(ty1 - ty0, tx1 - tx0), mode="bilinear", align_corners=False)
        crop = crop.to(donor_tokens.dtype).permute(0, 2, 3, 1)
        return crop.reshape(*prefix, (ty1 - ty0) * (tx1 - tx0), channels)

    def blend_target(self, tensor, alpha, channels=None, heads=None):
        # tensor [batch, tokens, channels] or [batch, heads, tokens, channels]
        if tensor.shape[0] != 4:
            raise RuntimeError(f"Expected CFG batch 4, got {tensor.shape[0]}")
        out = tensor.clone()
        target_idx = self.indices("target", tensor.device)
        weight = self.feather(tensor.device, tensor.dtype)
        if tensor.ndim == 3:
            target = out[2].index_select(0, target_idx)
            donor = self.aligned_donor(out[3, : self.grid_h * self.grid_w])
            if channels is None:
                target = target + alpha * weight * (donor - target)
            else:
                target[:, channels] = target[:, channels] + alpha * weight * (donor[:, channels] - target[:, channels])
            out[2].index_copy_(0, target_idx, target)
        elif tensor.ndim == 4:
            chosen = torch.arange(tensor.shape[1], device=tensor.device) if heads is None else heads
            target = out[2].index_select(0, chosen).index_select(1, target_idx)
            donor_all = self.aligned_donor(out[3, :, : self.grid_h * self.grid_w])
            donor = donor_all.index_select(0, chosen)
            target = target + alpha * weight.unsqueeze(0) * (donor - target)
            selected = out[2].index_select(0, chosen)
            selected.index_copy_(1, target_idx, target)
            out[2].index_copy_(0, chosen, selected)
        else:
            raise ValueError(tensor.shape)
        return out

    def top_channels(self, tensor, fraction):
        target_idx = self.indices("target", tensor.device)
        target = tensor[2].index_select(0, target_idx)
        donor = self.aligned_donor(tensor[3, : self.grid_h * self.grid_w])
        score = (donor.float() - target.float()).abs().mean(0)
        k = max(1, int(round(score.numel() * fraction)))
        return torch.topk(score, k=k).indices

    def mean_steer_target(self, tensor, alpha):
        if tensor.shape[0] != 4 or tensor.ndim != 3:
            raise RuntimeError(f"Expected [4,tokens,channels], got {tuple(tensor.shape)}")
        out = tensor.clone()
        target_idx = self.indices("target", tensor.device)
        donor_idx = self.indices("donor", tensor.device)
        target = out[2].index_select(0, target_idx)
        donor = out[3].index_select(0, donor_idx)
        direction = donor.float().mean(0) - target.float().mean(0)
        weight = self.feather(tensor.device, tensor.dtype)
        target = target + alpha * weight * direction.to(tensor.dtype).unsqueeze(0)
        out[2].index_copy_(0, target_idx, target)
        return out

    def modify_value(self, layer, value, image_tokens):
        if not self.active(layer) or self.config["mode"] != "v_patch":
            return value
        image = value[:, :, :image_tokens]
        patched = self.blend_target(image, self.config["alpha"])
        return torch.cat([patched, value[:, :, image_tokens:]], dim=2) if value.shape[2] > image_tokens else patched

    def modify_heads(self, layer, states, image_tokens):
        if not self.active(layer) or self.config["mode"] != "head_patch":
            return states
        heads = torch.as_tensor(self.config["heads"], dtype=torch.long, device=states.device)
        image = states[:, :, :image_tokens]
        patched = self.blend_target(image, self.config["alpha"], heads=heads)
        return torch.cat([patched, states[:, :, image_tokens:]], dim=2)

    def modify_attn_raw(self, layer, attn_raw):
        if not self.active(layer) or self.config["mode"] != "o_patch":
            return attn_raw
        channels = self.top_channels(attn_raw, self.config["fraction"])
        return self.blend_target(attn_raw, self.config["alpha"], channels=channels)

    def modify_attn_gated(self, layer, attn_gated):
        if not self.active(layer) or self.config["mode"] != "selective_residual":
            return attn_gated
        channels = self.top_channels(attn_gated, self.config["fraction"])
        return self.blend_target(attn_gated, self.config["alpha"], channels=channels)

    def modify_h_attn(self, layer, h_attn):
        if not self.active(layer) or self.config["mode"] not in ("reference_anchor", "anchor_gate"):
            return h_attn
        return self.mean_steer_target(h_attn, self.config["alpha"])

    def modify_f_gated(self, layer, f_gated):
        if not self.active(layer) or self.config["mode"] not in ("mlp_gate", "anchor_gate"):
            return f_gated
        out = f_gated.clone()
        idx = self.indices("target", out.device)
        region = out[2].index_select(0, idx) * self.config["beta"]
        out[2].index_copy_(0, idx, region)
        return out


class JointProcessor:
    def __init__(self, controller, layer):
        self.controller = controller
        self.layer = layer

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        batch_size = hidden_states.shape[0]
        image_tokens = hidden_states.shape[1]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
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
            query, key, value = torch.cat([query, cq], 2), torch.cat([key, ck], 2), torch.cat([value, cv], 2)
        value = self.controller.modify_value(self.layer, value, image_tokens)
        states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        states = self.controller.modify_heads(self.layer, states, image_tokens)
        combined = states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        if encoder_hidden_states is not None:
            combined, encoder_hidden_states = combined[:, :image_tokens], combined[:, image_tokens:]
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        combined = attn.to_out[1](attn.to_out[0](combined))
        return (combined, encoder_hidden_states) if encoder_hidden_states is not None else combined


def install(pipe, controller):
    for layer in controller.active_layers:
        block = pipe.transformer.transformer_blocks[layer]
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
            else:
                norm_ctx, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
            attn_raw, context_attn_output = self.attn(
                hidden_states=norm_h, encoder_hidden_states=norm_ctx, **joint_attention_kwargs
            )
            attn_raw = controller.modify_attn_raw(_layer, attn_raw)
            attn_gated = gate_msa.unsqueeze(1) * attn_raw
            if self.use_dual_attention:
                attn_gated = attn_gated + gate_msa2.unsqueeze(1) * self.attn2(hidden_states=norm_h2, **joint_attention_kwargs)
            attn_gated = controller.modify_attn_gated(_layer, attn_gated)
            h_attn = controller.modify_h_attn(_layer, h_in + attn_gated)
            x_norm = self.norm2(h_attn)
            x_mod = x_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            if self._chunk_size is not None:
                raise RuntimeError("Phase 2.8 requires feed-forward chunking disabled")
            f_gated = gate_mlp.unsqueeze(1) * self.ff(x_mod)
            h_out = h_attn + controller.modify_f_gated(_layer, f_gated)
            if self.context_pre_only:
                encoder_hidden_states = None
            else:
                encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
                norm_ctx2 = self.norm2_context(encoder_hidden_states)
                norm_ctx2 = norm_ctx2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_ctx2)
            return encoder_hidden_states, h_out

        block.forward = types.MethodType(forward, block)
    pipe.transformer.register_forward_pre_hook(controller.transformer_pre, with_kwargs=True)


def save_png(image, path, seed, prompt, latent_sha, config):
    info = PngInfo()
    info.add_text("seed", str(seed))
    info.add_text("prompt", prompt)
    info.add_text("latent_sha256", latent_sha)
    info.add_text("intervention", json.dumps(config, sort_keys=True))
    image.save(path, pnginfo=info)


def run_reference(pipe, controller, seed, args, run_root):
    reference_path = run_root / "reference" / f"seed{seed:06d}__B_only.png"
    if reference_path.exists():
        print(json.dumps({"event": "reference_skipped", "seed": seed}), flush=True)
        return
    latent, latent_sha = make_latent(pipe, seed, 1, args.height, args.width)
    controller.reset({"name": "canonical_B_only", "mode": "none", "steps": [], "layers": []})
    started = time.time()
    with torch.inference_mode():
        image = pipe(
            prompt=B_PROMPT,
            prompt_2=B_PROMPT,
            prompt_3=B_PROMPT,
            negative_prompt="",
            negative_prompt_2="",
            negative_prompt_3="",
            num_inference_steps=args.steps,
            guidance_scale=4.5,
            height=args.height,
            width=args.width,
            latents=latent,
            output_type="pil",
        ).images[0]
    save_png(image, reference_path, seed, B_PROMPT, latent_sha, {"mode": "canonical_B_only"})
    print(
        json.dumps(
            {
                "event": "reference_completed",
                "worker": args.worker,
                "seed": seed,
                "seconds": time.time() - started,
            }
        ),
        flush=True,
    )


def run_one(pipe, controller, config, seed, args, run_root):
    target_path = run_root / "combined" / f"seed{seed:06d}__{config['name']}.png"
    donor_path = run_root / "single_vase" / f"seed{seed:06d}__{config['name']}__B.png"
    if target_path.exists() and donor_path.exists() and not args.overwrite:
        print(json.dumps({"event": "skipped", "seed": seed, "name": config["name"]}), flush=True)
        return
    latent, latent_sha = make_latent(pipe, seed, 2, args.height, args.width)
    controller.reset(config)
    prompts = [AB_PROMPT, B_PROMPT]
    started = time.time()
    with torch.inference_mode():
        images = pipe(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            negative_prompt=["", ""],
            negative_prompt_2=["", ""],
            negative_prompt_3=["", ""],
            num_inference_steps=args.steps,
            guidance_scale=4.5,
            height=args.height,
            width=args.width,
            latents=latent,
            output_type="pil",
        ).images
    save_png(images[0], target_path, seed, AB_PROMPT, latent_sha, config)
    save_png(images[1], donor_path, seed, B_PROMPT, latent_sha, {"mode": "unchanged_B_only"})
    print(
        json.dumps(
            {
                "event": "completed",
                "worker": args.worker,
                "seed": seed,
                "name": config["name"],
                "seconds": time.time() - started,
                "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        ),
        flush=True,
    )


def main():
    args = parse_args()
    rows = catalog()
    if args.names:
        requested = set(args.names)
        rows = [row for row in rows if row["name"] in requested]
        missing = requested - {row["name"] for row in rows}
        if missing:
            raise ValueError(f"Unknown conditions: {sorted(missing)}")
    if args.smoke:
        args.steps, args.height, args.width = 2, 512, 512
        for row in rows:
            row["steps"], row["layers"] = [0, 1], [36]
        run_root = OUT / "smoke"
        active_steps, active_layers = (0, 1), (36,)
    else:
        run_root = OUT / "formal"
        active_steps, active_layers = FORMAL_STEPS, FORMAL_LAYERS
    for sub in ("combined", "single_vase", "reference", "logs", "analysis"):
        (run_root / sub).mkdir(parents=True, exist_ok=True)
    (run_root / "catalog.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    controller = Controller(active_steps, active_layers, args.height, args.width)
    install(pipe, controller)
    for seed in args.seeds:
        run_reference(pipe, controller, seed, args, run_root)
        for row in rows:
            run_one(pipe, controller, row, seed, args, run_root)


if __name__ == "__main__":
    main()
