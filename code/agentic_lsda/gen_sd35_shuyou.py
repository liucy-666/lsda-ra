"""Generate the cultural pair with SD3.5-Large (CPU offload, full 28 steps) for inspection."""
import torch
from diffusers import StableDiffusion3Pipeline

M = "/home/dell/models/sd3.5-large"
OUT = "/home/dell/Z1x/image_by_pry/_sd35_cultural.png"
prompt = ("Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
          "an Italian maiolica vase on the right; both fully visible, separate, and similar in size.")
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype=torch.float16, local_files_only=True)
pipe.enable_model_cpu_offload()
pipe.set_progress_bar_config(disable=True)
seed = 1011
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
gen = torch.Generator("cuda").manual_seed(seed)
img = pipe(prompt=prompt, generator=gen, num_inference_steps=28, guidance_scale=4.5,
           width=768, height=768).images[0]
img.save(OUT)
print("saved", OUT)
