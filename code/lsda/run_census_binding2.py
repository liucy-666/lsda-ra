"""Routing baseline v2: DreamRenderer-style binding with SAM entity tokens + mid-layer band.

Fixes the strawman issues of v1 (half-plane tokens + all-layer hard blocking):
  - token sets come from SAM entity masks on the native SS (not color half-planes)
  - hard blocking only on a middle layer band (17:26), soft elsewhere
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, "/science/wx/pry/EXP1/code")

import lsda_pipeline_v14_rect as L  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402
from diffusers import StableDiffusion3Pipeline  # noqa: E402

STEPS = 28
GUIDANCE = 4.5
SIZE = 1024


class BindProc:
    def __init__(self, orig, a_idx, b_idx):
        self.orig = orig
        self.a_idx = a_idx
        self.b_idx = b_idx
        self.mask = None

    def __call__(self, attn_self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
        b = hidden_states.shape[0]
        img_n = hidden_states.shape[1]
        heads = attn_self.heads
        hd = attn_self.to_q.out_features // heads
        q = attn_self.to_q(hidden_states).view(b, img_n, heads, hd).transpose(1, 2)
        k = attn_self.to_k(hidden_states).view(b, img_n, heads, hd).transpose(1, 2)
        v = attn_self.to_v(hidden_states).view(b, img_n, heads, hd).transpose(1, 2)
        if attn_self.norm_q is not None:
            q = attn_self.norm_q(q)
            k = attn_self.norm_k(k)
        if encoder_hidden_states is not None:
            en = encoder_hidden_states.shape[1]
            eq = attn_self.add_q_proj(encoder_hidden_states).view(b, en, heads, hd).transpose(1, 2)
            ek = attn_self.add_k_proj(encoder_hidden_states).view(b, en, heads, hd).transpose(1, 2)
            ev = attn_self.add_v_proj(encoder_hidden_states).view(b, en, heads, hd).transpose(1, 2)
            if attn_self.norm_added_q is not None:
                eq = attn_self.norm_added_q(eq)
                ek = attn_self.norm_added_k(ek)
            q = torch.cat([q, eq], 2)
            k = torch.cat([k, ek], 2)
            v = torch.cat([v, ev], 2)
        seq = q.shape[2]
        if self.mask is None or self.mask.device != q.device:
            m = torch.zeros(1, 1, seq, seq, device=q.device, dtype=q.dtype)
            ai = torch.tensor(self.a_idx, device=q.device)
            bi = torch.tensor(self.b_idx, device=q.device)
            m[:, :, ai[:, None], bi[None, :]] = -1e4
            m[:, :, bi[:, None], ai[None, :]] = -1e4
            self.mask = m
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask)
        out = out.transpose(1, 2).reshape(b, -1, heads * hd).to(q.dtype)
        hidden, enc_hidden = out[:, :img_n], out[:, img_n:]
        hidden = attn_self.to_out[0](hidden)
        if enc_hidden is not None and encoder_hidden_states is not None and not getattr(attn_self, "context_pre_only", False):
            enc_hidden = attn_self.to_add_out(enc_hidden)
        return (hidden, enc_hidden) if encoder_hidden_states is not None else hidden


def owner_tokens(owners, patch):
    grid = (SIZE // 8) // patch
    idx = []
    for owner in owners:
        m = np.asarray(Image.fromarray((owner.astype(np.uint8) * 255)).resize((grid, grid), Image.BILINEAR)) > 128
        idx.append(sorted(set(int(r * grid + c) for r, c in np.argwhere(m))))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1011, 1012, 1013])
    ap.add_argument("--layers", default="17:26")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--sam-model-dir", type=Path, required=True)
    ap.add_argument("--census-images", type=Path, default=None,
                    help="Reuse existing native SS PNGs instead of regenerating them.")
    args = ap.parse_args()

    from pairs100 import ss_prompt  # noqa: E402

    lo, hi = (int(x) for x in args.layers.split(":"))
    layers = list(range(lo, hi))
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    sam_processor = SamProcessor.from_pretrained(args.sam_model_dir, local_files_only=True)
    sam_model = SamModel.from_pretrained(args.sam_model_dir, local_files_only=True).to("cuda").eval()

    def gen(prompt, seed):
        g = torch.Generator("cuda").manual_seed(seed)
        return pipe(prompt=prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                    height=SIZE, width=SIZE, generator=g).images[0]

    patch = int(pipe.transformer.config.patch_size)
    for idx in args.pairs:
        prompt = ss_prompt(idx)
        for seed in args.seeds:
            out_png = args.out / f"binding2_p{idx:03d}_s{seed}.png"
            if out_png.exists():
                continue
            t0 = time.time()
            if args.census_images is not None:
                ss = Image.open(args.census_images / f"pair{idx:03d}_seed{seed}_SS.png").convert("RGB")
            else:
                ss = gen(prompt, seed)
            xs = [L.expected_x("left", 0, 2), L.expected_x("right", 1, 2)]
            cand_sets = [L.sam_candidates(sam_processor, sam_model, ss, x) for x in xs]
            owners, _, _ = L.clean_and_partition(L.choose_joint(cand_sets))
            a_idx, b_idx = owner_tokens(owners, patch)
            procs = []
            for i in layers:
                blk = pipe.transformer.transformer_blocks[i]
                proc = BindProc(blk.attn.processor, a_idx, b_idx)
                blk.attn.processor = proc
                procs.append(proc)
            bound = gen(prompt, seed)
            for i, proc in zip(layers, procs):
                pipe.transformer.transformer_blocks[i].attn.processor = proc.orig
            bound.save(out_png)
            (args.out / f"binding2_p{idx:03d}_s{seed}.json").write_text(
                json.dumps({"pair": idx, "seed": seed, "layers": args.layers,
                            "a_tokens": len(a_idx), "b_tokens": len(b_idx),
                            "seconds": round(time.time() - t0, 1)}, indent=1),
                encoding="utf-8")
            print(json.dumps({"done": out_png.name, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
