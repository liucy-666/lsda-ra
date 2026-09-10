"""Inspect SD3.5 transformer attention structure to design the hook."""
import sys
sys.path.insert(0, "/science/wx/pry/MMDIT/code/lsda")
from diffusers import StableDiffusion3Pipeline

M = "/science/wx/pry/models/stable-diffusion-3.5-large"
pipe = StableDiffusion3Pipeline.from_pretrained(M, torch_dtype="float16", local_files_only=True).to("cuda")
tr = pipe.transformer
print("transformer type:", type(tr).__name__)
print("num blocks:", len(tr.transformer_blocks))
b0 = tr.transformer_blocks[0]
print("block type:", type(b0).__name__)
print("block attrs:", [a for a in dir(b0) if not a.startswith('_')][:20])
attn = getattr(b0, "attn", None)
print("attn type:", type(attn).__name__ if attn else None)
if attn is not None:
    print("attn attrs:", [a for a in dir(attn) if not a.startswith('_')][:25])
    import inspect
    try:
        src = inspect.getsource(type(attn).forward)
        print("---- attn.forward source (first 2500 chars) ----")
        print(src[:2500])
    except Exception as e:
        print("source error:", e)
