"""Sweep native generations across seeds for a hard cultural pair to find a confused case."""
import json
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = Path(os.environ.get("SWEEP_OUT", "/home/dell/Z1x/image_by_pry/_confused"))
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
SEEDS = [int(x) for x in os.environ.get("SWEEP_SEEDS", "1011,3,7,11,15,17,23,29").split(",")]
PROMPT = os.environ.get(
    "SWEEP_PROMPT",
    "Neutral studio background: a Chinese blue-and-white porcelain vase on the left "
    "and an Italian maiolica blue-and-white vase on the right; both fully visible, "
    "separate, similar in size and shape.")
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


log("model ready")
for s in SEEDS:
    g = torch.Generator("cuda").manual_seed(s)
    img = pipe(prompt=PROMPT, generator=g, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE, width=SIZE, height=SIZE).images[0]
    img.save(OUT / f"native_{s}.png")
    log(f"seed {s} done")

(OUT / "meta.json").write_text(json.dumps(
    {"size": SIZE, "steps": STEPS, "guidance": GUIDANCE, "seeds": SEEDS, "prompt": PROMPT},
    indent=1), encoding="utf-8")
print(json.dumps({"event": "complete", "seeds": SEEDS}), flush=True)
