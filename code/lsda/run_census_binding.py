"""Batch routing-level baseline (DreamRenderer-style image attention binding) for the census.

For each (pair, seed): generate native SS, segment A/B regions from the SS image by
color foreground + x split, then regenerate with a cross-entity attention mask that
blocks A-image tokens <-> B-image tokens on a layer band. Output: binding.png + audit.
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

from diffusers import StableDiffusion3Pipeline  # noqa: E402

STEPS = 28
GUIDANCE = 4.5
SIZE = 1024


class BindProc:
    """Cross-entity attention mask on the concatenated [image | text] sequence."""

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


def segment_tokens(pipe, image: Image.Image):
    patch = int(pipe.transformer.config.patch_size)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    border = np.concatenate(
        [arr[:5].reshape(-1, 3), arr[-5:].reshape(-1, 3), arr[:, :5].reshape(-1, 3), arr[:, -5:].reshape(-1, 3)], 0
    )
    bg = np.median(border, 0)
    fg = np.linalg.norm(arr - bg[None, None], axis=-1) > 0.18
    ys, xs = np.where(fg)
    xmid = xs.mean() if len(xs) else 0.5 * SIZE
    left = fg.copy()
    left[:, int(xmid):] = False
    right = fg.copy()
    right[:, :int(xmid)] = False
    h_tok = w_tok = (SIZE // 8) // patch

    def to_tok(m):
        mm = np.asarray(Image.fromarray((m.astype(np.uint8) * 255)).resize((w_tok, h_tok), Image.BILINEAR)) > 128
        return np.argwhere(mm)

    a_idx = sorted(set(int(r * w_tok + c) for r, c in to_tok(left)))
    b_idx = sorted(set(int(r * w_tok + c) for r, c in to_tok(right)))
    return a_idx, b_idx, float(xmid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1011, 1012, 1013])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--layers", default="14:34")
    args = ap.parse_args()

    from pairs100 import ss_prompt  # local import after sys.path setup

    lo, hi = (int(x) for x in args.layers.split(":"))
    layers = list(range(lo, hi))
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_dir, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def gen(prompt, seed):
        g = torch.Generator("cuda").manual_seed(seed)
        return pipe(
            prompt=prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            height=SIZE, width=SIZE, generator=g,
        ).images[0]

    for idx in args.pairs:
        prompt = ss_prompt(idx)
        for seed in args.seeds:
            out_png = args.out / f"binding_p{idx:03d}_s{seed}.png"
            if out_png.exists():
                continue
            t0 = time.time()
            ss = gen(prompt, seed)
            a_idx, b_idx, xmid = segment_tokens(pipe, ss)
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
            (args.out / f"binding_p{idx:03d}_s{seed}.json").write_text(
                json.dumps(
                    {"pair": idx, "seed": seed, "layers": args.layers,
                     "a_tokens": len(a_idx), "b_tokens": len(b_idx), "x_mid": xmid,
                     "seconds": round(time.time() - t0, 1)},
                    ensure_ascii=False, indent=1,
                ),
                encoding="utf-8",
            )
            print(json.dumps({"done": out_png.name, "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
