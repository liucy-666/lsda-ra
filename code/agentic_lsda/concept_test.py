"""Single-instance concept test: does SD3.5 know Italian Mojolica/Maiolica when isolated?"""
import json
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = Path(os.environ.get("OUT", "/home/dell/Z1x/image_by_pry/_concept"))
SIZE = 512
STEPS = 28
GUIDANCE = 4.5
PROMPTS = {
    "mojolica_alone": "Studio photo of a single Italian Mojolica vase on a neutral background, fully visible, centered.",
    "maiolica_alone": "Studio photo of a single Italian Maiolica vase on a neutral background, fully visible, centered.",
    "bluewhite_alone": "Studio photo of a single Chinese blue-and-white porcelain vase on a neutral background, fully visible, centered.",
}
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()


def log(m):
    print(json.dumps({"log": m, "t": round(time.time() - t0, 1)}), flush=True)


pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
log("model ready")

for tag, pr in PROMPTS.items():
    g = torch.Generator("cuda").manual_seed(11)
    img = pipe(prompt=pr, generator=g, num_inference_steps=STEPS,
               guidance_scale=GUIDANCE, width=SIZE, height=SIZE).images[0]
    img.save(OUT / f"{tag}.png")
    log(f"{tag} done")

(OUT / "meta.json").write_text(json.dumps(PROMPTS, indent=1), encoding="utf-8")
print(json.dumps({"event": "complete"}), flush=True)
