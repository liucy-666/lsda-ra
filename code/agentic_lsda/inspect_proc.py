import sys
sys.path.insert(0, "/science/wx/pry/MMDIT/code/lsda")
from diffusers import StableDiffusion3Pipeline
M = "/science/wx/pry/models/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype="float16", local_files_only=True).to("cuda")
b0 = pipe.transformer.transformer_blocks[0]
attn = b0.attn
print("processor type:", type(attn.processor).__name__)
import inspect
try:
    src = inspect.getsource(type(attn.processor).__call__)
    print("---- processor.__call__ source (first 2000) ----")
    print(src[:2000])
except Exception as e:
    print("err", e)
try:
    print("attn processors available:", [n for n in dir(attn) if 'processor' in n.lower()][:10])
except Exception as e:
    print(e)
