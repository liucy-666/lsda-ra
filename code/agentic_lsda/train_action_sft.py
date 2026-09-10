"""Build + train action-head (timing planner) SFT on the 12-sample pilot grid.

Oracle labels from v6 drift scores: per (sample, target) best step = argmin
branch drift_p vs native (margin 0.03); none if no branch beats native.
Input: native final image + scene prompt + target; Output JSON action.
Small-sample smoke test: validate pipeline end to end.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/root/sft_data/manifests")

MARGIN = 0.03
STEPS = [0, 4, 8, 12, 16, 20, 24]
SCORES = "/root/sft_data/pilot_drift_scores.json"
MANIFEST = Path("/root/sft_data/manifests/counterfactual_pilot.json")
SS_DIR = "/root/sft_data/SS/2026_8_25_EXP_1"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUT = Path("/root/sft_out_action")

scores = json.loads(Path(SCORES).read_text(encoding="utf-8"))
tasks = json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]
by_sample = {}
for t in tasks:
    sid = t["sample_id"]
    by_sample.setdefault(sid, {"prompts": t, "native": None, "A": {}, "B": {}, "AB": {}})
    s = by_sample[sid]
    act = t["action"]
    if act["kind"] == "wait":
        s["native"] = scores[t["replay_id"]]["drift_p"]
    else:
        s[act["target"]][act["intervention_step"]] = scores[t["replay_id"]]["drift_p"]

records = []
for sid, s in sorted(by_sample.items()):
    nat = s["native"]
    for tgt in ("A", "B", "AB"):
        if nat is None or not s[tgt]:
            continue
        best = min(s[tgt], key=s[tgt].get)
        bp = s[tgt][best]
        if bp <= nat - MARGIN:
            action = {"action": "invoke", "step": int(best)}
        else:
            action = {"action": "none"}
        records.append({
            "sample": sid, "pair": s["prompts"]["pair_id"], "target": tgt,
            "image": f"{SS_DIR}/{sid}.jpg",
            "global_prompt": s["prompts"]["global_prompt"],
            "entity_A": s["prompts"]["entity_A_prompt"],
            "entity_B": s["prompts"]["entity_B_prompt"],
            "answer": json.dumps(action),
        })
print(f"action records: {len(records)}")

# ---- train/eval split by sample ----
import random
samples = sorted({r["sample"] for r in records})
random.Random(0).shuffle(samples)
eval_samples = set(samples[:3])
train = [r for r in records if r["sample"] not in eval_samples]
evals = [r for r in records if r["sample"] in eval_samples]
print(f"train {len(train)} (samples {len(samples)-3}) | eval {len(evals)} (samples {sorted(eval_samples)})")
for r in evals:
    print(f"  EVAL {r['sample']}[{r['target']}] gold={r['answer']}")

# ---- model setup (fresh base; smoke test) ----
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, quantization_config=quant,
                                                    device_map="auto", trust_remote_code=True)
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_disable()
model.config.use_cache = False
lora = LoraConfig(r=16, lora_alpha=32,
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
                  lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(model, lora)

# cache resized images
IMG = 448
imgs = {}
for rec in records:
    if rec["image"] not in imgs:
        imgs[rec["image"]] = Image.open(rec["image"]).convert("RGB").resize((IMG, IMG), Image.LANCZOS)

def build_text(rec):
    q = (f'One SD3.5 generation may have cultural drift. Scene: "{rec["global_prompt"]}". '
         f"Left should be: {rec['entity_A']}. Right should be: {rec['entity_B']}. "
         f"Target entity to repair: {rec['target']}. "
         f"Decide whether to invoke the LSDA local rediffusion expert for the target entity, "
         f"and if so at which diffusion step from {STEPS}. "
         'Answer JSON only: {"action": "invoke|none", "step": 0 or 4 or 8 or 12 or 16 or 20 or 24}')
    return q

def encode(text, image):
    conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": text},
    ]}]
    prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    full = prompt + '<|im_end|>'
    enc = processor(text=full, images=image, return_tensors="pt")
    prompt_enc = processor(text=prompt, images=image, return_tensors="pt")
    plen = prompt_enc["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[0, :plen] = -100
    return {k: v[0] for k, v in enc.items() if k != "pixel_values"} | {"pixel_values": enc["pixel_values"],
                                                                        "labels": labels}

# ---- training loop (tiny: many epochs) ----
import torch.optim as optim
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
scaler = torch.cuda.amp.GradScaler()
EPOCHS = 30
model.train()
step = 0
for epoch in range(EPOCHS):
    order = list(range(len(train)))
    random.Random(epoch).shuffle(order)
    tot = 0.0
    for idx in order:
        rec = train[idx]
        text = build_text(rec)
        # labels: answer after assistant marker (target only)
        q_conv = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": text},
        ]}]
        q_text = processor.apply_chat_template(q_conv, tokenize=False, add_generation_prompt=True)
        full_text = q_text + rec["answer"] + "<|im_end|>"
        img = imgs[rec["image"]]
        enc = processor(text=full_text, images=img, return_tensors="pt")
        q_enc = processor(text=q_text, images=img, return_tensors="pt")
        plen = q_enc["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :plen] = -100
        batch = {k: v.to("cuda") for k, v in enc.items()}
        batch["labels"] = labels.to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        loss = out.loss
        scaler.scale(loss).backward()
        tot += loss.item()
        if (idx + 1) % 8 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch} loss {tot / len(train):.4f}", flush=True)

# ---- eval ----
model.eval()
import re
def parse(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0:
        return {}
    try:
        return json.loads(text[s:e + 1])
    except Exception:
        return {}

def pred(rec):
    q_conv = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": build_text(rec)},
    ]}]
    q_text = processor.apply_chat_template(q_conv, tokenize=False, add_generation_prompt=True)
    img = imgs[rec["image"]]
    enc = processor(text=q_text, images=img, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=24, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
    raw = processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse(raw), raw

print("\n=== held-out eval ===")
for rec in evals:
    p, raw = pred(rec)
    gold = json.loads(rec["answer"])
    ok = p.get("action") == gold.get("action") and (
        gold.get("action") == "none" or p.get("step") == gold.get("step"))
    print(f"{rec['sample']}[{rec['target']}] gold={rec['answer']} pred={json.dumps(p)} ok={ok} raw={raw[:60]!r}")

model.save_pretrained(OUT / "lora")
print(f"DONE: {OUT / 'lora'}")
