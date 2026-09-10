import inspect
import sys
sys.path.insert(0, "/science/wx/pry/MMDIT/code/lsda")
from diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "/science/wx/pry/models/stable-diffusion-3.5-large",
    torch_dtype="float16", local_files_only=True).to("cuda")
pro = pipe.transformer.transformer_blocks[0].attn.processor
src = inspect.getsource(type(pro).__call__)
print(src[1500:4000])
