import importlib.util as u
from diffusers import StableDiffusion3Pipeline
import transformers
print("transformers", transformers.__version__)
print("cv2", bool(u.find_spec("cv2")), "scipy", bool(u.find_spec("scipy")),
      "skimage", bool(u.find_spec("skimage")))
try:
    import torch
    print("torch", torch.__version__)
except Exception as e:
    print("torch err", e)
