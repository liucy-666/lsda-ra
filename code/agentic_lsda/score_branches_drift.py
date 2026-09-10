"""Score pilot branch images with the v5 drift diagnoser (logits P(drift=1)).

For each task in counterfactual_pilot.json: forward with drift-task prompt +
prefix '{"drift": ' and take softmax P(token "1") at the last position.
Outputs JSON: replay_id -> {sample, step, target, drift_p}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/root/sft_data/manifests")
from train_sft import TASK_TMPL  # noqa: E402
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

MANIFEST = Path("/root/sft_data/manifests/counterfactual_pilot.json")
IMAGES = Path("/root/sft_data/pilot_images")
ADAPTER = "/root/sft_out_v6/lora"
OUT = Path("/root/sft_data/pilot_drift_scores.json")

payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
tasks = payload["tasks"]
print(f"tasks: {len(tasks)}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto", torch_dtype=torch.float16,
    trust_remote_code=True)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

tok1 = processor.tokenizer.convert_tokens_to_ids("1")

def find_image(rid):
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES / f"{rid}{ext}"
        if p.exists():
            return p
    return None

def drift_p(img_path, task):
    user = TASK_TMPL["drift"]["user"].format(
        global_prompt=task["global_prompt"],
        entity_A_prompt=task["entity_A_prompt"],
        entity_B_prompt=task["entity_B_prompt"],
    )
    conv = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": user},
    ]}]
    prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    prefix = prompt + '{"drift": '
    img = Image.open(img_path).convert("RGB").resize((448, 448), Image.LANCZOS)
    enc = processor(text=prefix, images=img, return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    return torch.softmax(logits.float(), dim=-1)[tok1].item()

results = {}
with torch.no_grad():
    for i, task in enumerate(tasks):
        rid = task["replay_id"]
        img = find_image(rid)
        if img is None:
            print(f"missing image: {rid}")
            continue
        results[rid] = {
            "sample": task["sample_id"],
            "step": task["action"].get("intervention_step"),
            "target": task["action"].get("target"),
            "kind": task["action"].get("kind"),
            "drift_p": drift_p(img, task),
        }
        if (i + 1) % 50 == 0:
            print(f"scored {i + 1}/{len(tasks)}", flush=True)

OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"saved {len(results)} scores -> {OUT}")
