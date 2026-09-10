"""Evaluate trained LoRA on held-out pairs: per-task accuracy."""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/root/sft_data/manifests")
from train_sft import TASK_TMPL  # noqa: E402

TASK_KEYS = {"drift": "drift", "leakage": "leakage", "accept": "action"}
TASKS = ["drift", "leakage", "accept"]


def parse_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(), flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--img-size", type=int, default=448)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # build per-task records with explicit task label
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    records = []
    for row in rows:
        for task in TASKS:
            tmpl = TASK_TMPL[task]
            if task == "accept" and (row["condition"] != "lsda_clean" or not row["task3_accept_reject"]):
                continue
            if task == "leakage" and row["task2_leakage_direction_qwen"] in (None, "unknown"):
                continue
            if task == "drift" and row["drift_label"] is None:
                continue
            gold = tmpl["label"](row)
            gold_val = json.loads(gold)[TASK_KEYS[task]]
            records.append({
                "image": row["image"], "pair": row["pair_id"], "task": task,
                "user": tmpl["user"].format(**row), "gold": str(gold_val).strip().lower(),
            })
    if args.limit:
        records = records[: args.limit]
    pairs = sorted({r["pair"] for r in records})
    eval_pairs = set(pairs[-20:])
    evals = [r for r in records if r["pair"] in eval_pairs]
    print(f"eval records: {len(evals)} (pairs: {len(eval_pairs)})")

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    results = []
    inc_path = "/root/sft_eval_incremental.jsonl"
    inc = open(inc_path, "w", encoding="utf-8")
    with torch.no_grad():
        for idx, rec in enumerate(evals):
            conv = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": rec["user"]},
            ]}]
            text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            img = Image.open(rec["image"]).convert("RGB").resize(
                (args.img_size, args.img_size), Image.LANCZOS
            )
            enc = processor(text=text, images=img, return_tensors="pt")
            enc = {k: v.to("cuda") for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=32, do_sample=False,
                                 pad_token_id=processor.tokenizer.pad_token_id)
            raw = processor.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = str(parse_json(raw).get(TASK_KEYS[rec["task"]], "")).strip().lower()
            ok = pred == rec["gold"]
            stats[rec["task"]]["total"] += 1
            stats[rec["task"]]["correct"] += int(ok)
            row = {"image": rec["image"], "task": rec["task"], "gold": rec["gold"],
                   "pred": pred, "ok": ok, "raw": raw}
            results.append(row)
            inc.write(json.dumps(row, ensure_ascii=False) + "\n")
            inc.flush()
            if (idx + 1) % 50 == 0:
                line = f"eval {idx + 1}/{len(evals)} | " + " ".join(
                    f"{t}={s['correct']}/{s['total']}({s['correct'] / max(s['total'], 1):.2f})"
                    for t, s in stats.items()
                )
                print(line, flush=True)
    inc.close()

    print("=" * 60)
    for task in TASKS:
        s = stats[task]
        acc = s["correct"] / max(s["total"], 1)
        print(f"{task:8s} accuracy: {acc:.3f} ({s['correct']}/{s['total']})")
    # per-class metrics for drift (guard against majority-class cheating)
    drift_rows = [r for r in results if r["task"] == "drift"]
    if drift_rows:
        tp = sum(1 for r in drift_rows if r["gold"] == "1" and r["pred"] == "1")
        fn = sum(1 for r in drift_rows if r["gold"] == "1" and r["pred"] != "1")
        fp = sum(1 for r in drift_rows if r["gold"] != "1" and r["pred"] == "1")
        tn = sum(1 for r in drift_rows if r["gold"] != "1" and r["pred"] != "1")
        recall = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        print(f"drift per-class: recall(1)={recall:.3f} precision(1)={prec:.3f} F1={f1:.3f} "
              f"| tp={tp} fn={fn} fp={fp} tn={tn}")
        neg_rate = sum(1 for r in drift_rows if r["gold"] == "0") / max(len(drift_rows), 1)
        print(f"drift majority-baseline accuracy (all-0): {neg_rate:.3f}")
    overall = sum(s["correct"] for s in stats.values()) / max(sum(s["total"] for s in stats.values()), 1)
    print(f"overall accuracy: {overall:.3f}")
    with open("/root/sft_eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"stats": {k: dict(v) for k, v in stats.items()}, "overall": overall,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("saved /root/sft_eval_results.json")


if __name__ == "__main__":
    main()
