"""Verify SD3.5 loads + generates on A100_shuyou_1 without OOM; check SAM."""
import sys, gc, time
from pathlib import Path
import torch

M = "/home/dell/models/sd3.5-large"
pipe = None
try:
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
    # model_cpu_offload like the local flux.py
    pipe.enable_model_cpu_offload()
    print("loaded SD3.5, device:", next(pipe.transformer.parameters()).device, flush=True)
except Exception as e:
    print("load err:", type(e).__name__, str(e)[:200], flush=True)
    sys.exit(1)

torch.manual_seed(1011)
try:
    img = pipe("Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
               "an Italian maiolica vase on the right; both fully visible, separate, similar in size.",
               num_inference_steps=8, guidance_scale=4.5, width=512, height=512).images[0]
    img.save("/home/dell/Z1x/image_by_pry/_test_sd35.png")
    print("GENERATION OK", flush=True)
except Exception as e:
    print("gen err:", type(e).__name__, str(e)[:250], flush=True)

# SAM check
import subprocess
for p in ["/home/dell/Z1x/models", "/home/dell/models", "/home/dell/Z1x/grounded_sam2"]:
    print(f"--- {p} ---")
    try:
        print(subprocess.run(["find", p, "-maxdepth", "3", "-iname", "*sam*", "-type", "d"],
                             capture_output=True, text=True, timeout=20).stdout[:300])
    except Exception as e:
        print(" ".join(map(str, e.args))[:80])
