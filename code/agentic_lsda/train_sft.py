"""SFT: fine-tune Qwen2.5-VL-3B-Instruct (QLoRA) on 3 diagnostic tasks.

T1 drift classification / T2 leakage direction / T3 accept-reject.
Fixed image size (448x448) + torch Dataset + custom collator for stable batching.
Split by pair_id. Run on the 4090 server.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
import torch.nn.functional as F

TASK_TMPL = {
    "drift": {
        "user": "Look at this AI-generated image. The intended scene is: \"{global_prompt}\". "
                "Left should be: {entity_A_prompt}. Right should be: {entity_B_prompt}. "
                "Decide whether cultural attribute drift occurred (any object's surface culture "
                "mismatches its intended culture). Answer JSON only: {{\"drift\": 0 or 1}}",
        "label": lambda r: json.dumps({"drift": r["drift_label"]}),
    },
    "leakage": {
        "user": "Look at this image (scene: \"{global_prompt}\"; left={entity_A_prompt}, "
                "right={entity_B_prompt}). Which direction did culture leak? "
                "Answer JSON only: {{\"leakage\": \"none|A_to_B|B_to_A|bidirectional\"}}",
        "label": lambda r: json.dumps({"leakage": r["task2_leakage_direction_qwen"]}),
    },
    "accept": {
        "user": "This image is a local-repair result (LSDA) for the scene \"{global_prompt}\" "
                "(left={entity_A_prompt}, right={entity_B_prompt}). Compared to the un-repaired "
                "image: did repair fix the drift (accept), damage it (reject), or was no repair "
                "needed (no_op)? Answer JSON only: {{\"action\": \"accept|reject|no_op\"}}",
        "label": lambda r: json.dumps({"action": r["task3_accept_reject"]}),
    },
}


def build_records(manifest: Path, tasks: list[str]) -> list[dict]:
    records = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for task in tasks:
            tmpl = TASK_TMPL[task]
            if task == "accept" and (row["condition"] != "lsda_clean" or not row["task3_accept_reject"]):
                continue
            if task == "leakage" and row["task2_leakage_direction_qwen"] in (None, "unknown"):
                continue
            if task == "drift" and row["drift_label"] is None:
                continue
            records.append({
                "image": row["image"],
                "user": tmpl["user"].format(**row),
                "answer": tmpl["label"](row),
                "pair_id": row["pair_id"],
            })
    return records


def build_balanced_records(manifest: Path, tasks: list[str], seed: int = 7) -> list[dict]:
    """Balanced + reduced training set:
    - drift: all positives + equal # negatives sampled (50:50)
    - leakage: all non-none rows (kept as-is)
    - accept: all accept/reject + no_op downsampled to match accept count
    """
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    out: list[dict] = []

    def rec(row, task, tmpl):
        return {
            "image": row["image"],
            "user": tmpl["user"].format(**row),
            "answer": tmpl["label"](row),
            "pair_id": row["pair_id"],
        }

    if "drift" in tasks:
        pos = [r for r in rows if r["drift_label"] == 1]
        neg = [r for r in rows if r["drift_label"] == 0]
        rng.shuffle(neg)
        out += [rec(r, "drift", TASK_TMPL["drift"]) for r in pos]
        out += [rec(r, "drift", TASK_TMPL["drift"]) for r in neg[: len(pos)]]
        print(f"[balance] drift: pos={len(pos)} neg_kept={min(len(neg), len(pos))}")

    if "leakage" in tasks:
        directional = [r for r in rows if r["task2_leakage_direction_qwen"] not in (None, "unknown", "none")]
        none_rows = [r for r in rows if r["task2_leakage_direction_qwen"] == "none"]
        rng.shuffle(none_rows)
        keep = directional + none_rows[: max(len(directional), 1)]
        out += [rec(r, "leakage", TASK_TMPL["leakage"]) for r in keep]
        print(f"[balance] leakage: directional={len(directional)} none_kept={min(len(none_rows), len(directional))}")

    if "accept" in tasks:
        acc = [r for r in rows if r["condition"] == "lsda_clean" and r["task3_accept_reject"] == "accept"]
        rej = [r for r in rows if r["condition"] == "lsda_clean" and r["task3_accept_reject"] == "reject"]
        noop = [r for r in rows if r["condition"] == "lsda_clean" and r["task3_accept_reject"] == "no_op"]
        rng.shuffle(noop)
        out += [rec(r, "accept", TASK_TMPL["accept"]) for r in acc + rej + noop[: len(acc)]]
        print(f"[balance] accept: acc={len(acc)} rej={len(rej)} noop_kept={min(len(noop), len(acc))}")
    return out


class VLDS(torch.utils.data.Dataset):
    def __init__(self, records, processor, img_size: int):
        self.records = records
        self.processor = processor
        self.img_size = img_size
        # pre-resize all images once (RAM cache) to keep the step loop fast
        self.images = []
        for rec in records:
            img = Image.open(rec["image"]).convert("RGB").resize(
                (img_size, img_size), Image.LANCZOS
            )
            self.images.append(img)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        conv = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": rec["user"]},
        ]}]
        prompt = self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        full_text = prompt + rec["answer"] + "<|im_end|>"
        img = self.images[i]
        enc = self.processor(text=full_text, images=img, return_tensors="pt")
        # mask prompt positions in labels; keep only the assistant answer.
        # Must encode the prompt WITH the image so the image_pad expansion
        # matches the full-text encoding (tokenizer(prompt) alone under-counts).
        prompt_enc = self.processor(text=prompt, images=img, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = enc["input_ids"].clone()
        labels[0, :prompt_len] = -100
        enc["labels"] = labels
        out = {}
        for k, v in enc.items():
            if k == "pixel_values":
                out[k] = v  # (num_patches, C)
            else:
                out[k] = v[0]
        return out


class VLCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_ids = [f["input_ids"].tolist() for f in features]
        attention_mask = [f["attention_mask"].tolist() for f in features]
        labels = [f["labels"].tolist() for f in features]
        batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            return_tensors="pt", padding=True,
        )
        max_len = batch["input_ids"].shape[1]
        batch["labels"] = torch.full((len(features), max_len), -100, dtype=batch["input_ids"].dtype)
        for i, lab in enumerate(labels):
            batch["labels"][i, : len(lab)] = torch.tensor(lab, dtype=batch["labels"].dtype)
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.stack([f["image_grid_thw"] for f in features])
        return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", type=Path, default="/root/sft_out")
    ap.add_argument("--tasks", default="drift,leakage")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--eval-pairs", type=int, default=20)
    ap.add_argument("--no-balance", action="store_true", help="use full unbalanced records")
    args = ap.parse_args()

    if args.no_balance:
        train_records_all = build_records(args.manifest, args.tasks.split(","))
    else:
        train_records_all = build_balanced_records(args.manifest, args.tasks.split(","))
    # eval set: FULL unbalanced records (real deployment distribution)
    eval_records_all = build_records(args.manifest, args.tasks.split(","))

    def validate(recs):
        out = []
        for rec in recs:
            try:
                Image.open(rec["image"]).verify()
                out.append(rec)
            except Exception:
                pass
        return out

    train_records_all = validate(train_records_all)
    eval_records_all = validate(eval_records_all)

    # split by pair: eval pairs held out from BOTH train and eval construction
    eval_pairs = set(sorted({r["pair_id"] for r in eval_records_all})[-args.eval_pairs:])
    train_records = [r for r in train_records_all if r["pair_id"] not in eval_pairs]
    eval_records = [r for r in eval_records_all if r["pair_id"] in eval_pairs]
    print(f"train={len(train_records)} eval={len(eval_records)} eval_pairs={len(eval_pairs)}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = VLDS(train_records, processor, args.img_size)
    eval_ds = VLDS(eval_records, processor, args.img_size)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- manual training loop (deterministic; avoids Trainer data-pipeline issues) ----
    import torch.optim as optim

    collator = VLCollator(processor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler()
    accum = 8
    steps_per_epoch = (len(train_ds) + accum - 1) // accum
    total_steps = steps_per_epoch * args.epochs
    print(f"train={len(train_ds)} eval={len(eval_ds)} steps/epoch={steps_per_epoch} total={total_steps}")

    def run_batch(ds, indices):
        feats = [ds[i] for i in indices]
        return collator(feats)

    def evaluate():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(eval_ds), 4):
                batch = run_batch(eval_ds, list(range(i, min(i + 4, len(eval_ds)))))
                batch = {k: v.to("cuda") for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    out = model(**batch)
                if out.loss is not None and not torch.isnan(out.loss):
                    total += out.loss.item() * len(batch["input_ids"])
                    count += len(batch["input_ids"])
        model.train()
        return total / max(count, 1)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(train_ds)))
        random.Random(epoch).shuffle(order)
        opt_accum = 0.0
        for i, idx in enumerate(order):
            batch = run_batch(train_ds, [idx])
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**batch)
            loss = out.loss / accum
            scaler.scale(loss).backward()
            opt_accum += out.loss.item()
            if (i + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {opt_accum / accum:.4f}", flush=True)
                opt_accum = 0.0
        if opt_accum:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        eval_loss = evaluate()
        print(f"epoch {epoch} eval_loss {eval_loss:.4f}", flush=True)

    model.save_pretrained(args.out / "lora")
    processor.save_pretrained(args.out / "lora")
    print(f"DONE: {args.out / 'lora'}")


if __name__ == "__main__":
    main()
