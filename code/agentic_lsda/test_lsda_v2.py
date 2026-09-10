"""Test LSDA V2 design on one pair (Chinese blue-white vase vs Italian maiolica vase).

Outputs: native SS, standalone A, standalone B, LSDA v1, LSDA v2 + comparison.jpg.
V2: region (right/maiolica) re-noised to full noise, denoised n steps with its own
prompt, merged at step n (other regions + background stay on native), then continue
denoising the whole latent with the FULL SS prompt to the end.
Run on A100 GPU (SD3.5 + sam-vit-base + code/lsda).
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/science/wx/pry/MMDIT/code/lsda")
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from transformers import SamModel, SamProcessor
import lsda_pipeline as L
from runtime import prepare_schedule, transformer_pair, region_cfg_score, encode_prompts

MODEL_DIR = "/science/wx/pry/models/stable-diffusion-3.5-large"
SAM_DIR = "/science/wx/pry/models/sam-vit-base"
OUT = Path("/science/wx/pry/MMDIT/exp_v2_test")
SEED = 1011
SIZE = 768  # lowered to fit SD3.5-Large 1024-CFG peak memory; comparison res adjusted
STEPS = 28
GUIDANCE = 4.5
V2_STEP = 8  # detection/rejoin step for V2

SS = ("Neutral studio background: a Chinese blue-and-white porcelain vase on the left, "
      "an Italian maiolica vase on the right; both fully visible, separate, and similar in size.")
PA = "a Chinese blue-and-white porcelain vase"
PB = "an Italian maiolica vase"

OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

def log(msg):
    print(json.dumps({"log": msg, "t": round(time.time() - t0, 1)}), flush=True)

pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, local_files_only=True).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()  # cut transformer attention peak memory
pipe.enable_vae_slicing()
helpers = (prepare_schedule, transformer_pair, region_cfg_score)
patch = int(pipe.transformer.config.patch_size)
initial, sha = L.make_latent(pipe, SEED)
log("model+latent ready")

def cfg_full(enc, index, latent, timestep):
    e = torch.cat([enc["negative"], enc["positive"][index:index + 1]], dim=0)
    p = torch.cat([enc["negative_pooled"], enc["positive_pooled"][index:index + 1]], dim=0)
    uncond, cond = transformer_pair(pipe, latent, timestep, e, p).chunk(2)
    return uncond + GUIDANCE * (cond - uncond)

def full_denoise(enc, start_latent, start=0, end=None, index=0):
    end = end if end is not None else STEPS
    ts, mu = prepare_schedule(pipe, start_latent, STEPS)
    sched = copy.deepcopy(pipe.scheduler)
    lat = start_latent.clone()
    for i in range(start, end):
        v = cfg_full(enc, index, lat, ts[i])
        lat = sched.step(v.to(lat.dtype), ts[i], lat, return_dict=False)[0].to(start_latent.dtype)
    return lat, ts, mu

# 1) native SS trajectory -> save ss + states
enc_ss = encode_prompts(pipe, (SS, PA, PB))
native_states, _, mu = L.native_ss_trajectory(pipe, initial, enc_ss, helpers)
ss = L.decode(pipe, native_states[-1]); ss.save(OUT / "ss.png"); log("ss done")

# 2) SAM segment left/right
sam_processor = SamProcessor.from_pretrained(SAM_DIR, local_files_only=True)
sam_model = SamModel.from_pretrained(SAM_DIR, local_files_only=True).to("cuda").eval()
candsA = L.sam_candidates(sam_processor, sam_model, ss, L.expected_x("left", 0, 2))
candsB = L.sam_candidates(sam_processor, sam_model, ss, L.expected_x("right", 1, 2))
chosen = L.choose_joint([candsA, candsB])
owners_img, bg_img, seg_diag = L.clean_and_partition(chosen)
owners, background = L.image_masks_to_latent(owners_img, initial)
del sam_model, sam_processor; torch.cuda.empty_cache()
log("sam done")

# 3) standalone A / B
sa, _, _ = full_denoise(encode_prompts(pipe, (PA,)), initial, index=0)
L.decode(pipe, sa).save(OUT / "standalone_A.png"); log("standalone_A done")
sb, _, _ = full_denoise(encode_prompts(pipe, (PB,)), initial, index=0)
L.decode(pipe, sb).save(OUT / "standalone_B.png"); log("standalone_B done")

# 4) LSDA v1 (denoise_specialists: experts from step 0)
lv1, _ = L.denoise_specialists(pipe, initial, enc_ss, owners, background, native_states, helpers)
L.decode(pipe, lv1).save(OUT / "lsda_v1.png"); log("lsda_v1 done")

# 5) LSDA v2: rewind right region (B) to full noise, denoise V2_STEP steps, merge, continue SS
enc_B = encode_prompts(pipe, (PB,))
gts, gmu = prepare_schedule(pipe, initial, STEPS)  # GLOBAL schedule (for full/crop consistency)
bbox = L.owner_bounds(owners[1], patch)
l, t_, r, b = bbox
crop = initial[:, :, t_:b, l:r].clone()
sched = copy.deepcopy(pipe.scheduler)
for i in range(V2_STEP):
    sc = region_cfg_score(pipe, crop, gts[i], enc_B["negative"], enc_B["positive"][0:1],
                          enc_B["negative_pooled"], enc_B["positive_pooled"][0:1], GUIDANCE)
    crop = sched.step(sc.to(crop.dtype), gts[i], crop, return_dict=False)[0].to(initial.dtype)
log("v2 region re-rolldone")
# composite at step n: native state right before step n
nat_n = native_states[V2_STEP - 1].clone() if V2_STEP > 0 else initial.clone()
merged = background * nat_n
merged = merged + owners[0] * nat_n
merged_c = merged.clone()
merged_c[:, :, t_:b, l:r] = owners[1][:, :, t_:b, l:r] * crop
merged = merged_c
# continue with FULL SS prompt from step n
lv2, _, _ = full_denoise(enc_ss, merged, start=V2_STEP, end=STEPS, index=0)
L.decode(pipe, lv2).save(OUT / "lsda_v2.png"); log("lsda_v2 done")

# 6) comparison figure
def panel(img):
    return img.resize((SIZE, SIZE), Image.LANCZOS)
imgs = [("Native SS", ss), ("StandAlone A", L.decode(pipe, sa)), ("StandAlone B", L.decode(pipe, sb)),
        ("LSDA v1", L.decode(pipe, lv1)), ("LSDA v2", L.decode(pipe, lv2))]
W = SIZE * len(imgs)
canvas = Image.new("RGB", (W, SIZE + 46), (250, 246, 238))
from PIL import ImageDraw
d = ImageDraw.Draw(canvas)
for i, (lab, im) in enumerate(imgs):
    canvas.paste(panel(im), (i * SIZE, 46)); d.text((i * SIZE + 16, 14), lab, fill=(31, 78, 121))
canvas.save(OUT / "comparison.jpg", quality=95)
log("comparison done")

record = {"seed": SEED, "ss_prompt": SS, "A": PA, "B": PB, "v1_experts_from": 0,
          "v2_rewind_step": 0, "v2_rejoin_step": V2_STEP,
          "segmentation_owner_fractions": seg_diag.get("owner_fractions_image"),
          "latent_sha": sha, "v2_bbox": bbox}
(OUT / "audit.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"event": "complete", "out": str(OUT), "t": time.time() - t0}), flush=True)
