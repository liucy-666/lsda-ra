from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path("/root/private_data/Pry")
MODEL_DIR = ROOT / "models" / "stable-diffusion-3.5-large"
RUN_DIR = ROOT / "Experience" / "phase1"
RESULTS_DIR = RUN_DIR / "results"

CONDITIONS = {
    "SS": {
        "chinese": "Chinese blue-and-white porcelain plate",
        "italian": "Italian maiolica vase",
    },
    "LS": {
        "chinese": (
            "Chinese plate which presents delicate blue patterns painted beneath transparent glaze "
            "against pure white porcelain base, only blue and white in color"
        ),
        "italian": "Italian maiolica vase",
    },
    "SL": {
        "chinese": "Chinese blue-and-white porcelain plate",
        "italian": (
            "Italian vase which has a milky white tin-glazed ceramic surface decorated with bright "
            "multicolored painted motifs"
        ),
    },
    "LL": {
        "chinese": (
            "Chinese plate which has a pure white porcelain surface decorated with delicate cobalt-blue "
            "painted motifs beneath a transparent glaze, using only blue and white colors"
        ),
        "italian": (
            "Italian vase which has a milky white tin-glazed ceramic surface decorated with bright "
            "multicolored painted motifs in blue, yellow, green, and orange colors"
        ),
    },
}


def prompt_record(condition: str, order: str) -> dict:
    desc = CONDITIONS[condition]
    chinese = desc["chinese"]
    italian = desc["italian"]
    if order == "forward":
        prompt = f"A {chinese} is next to an {italian}."
    elif order == "reverse":
        prompt = f"An {italian} is next to a {chinese}."
    else:
        raise ValueError(order)
    return {
        "condition": condition,
        "prompt_order": order,
        "prompt": prompt,
        "chinese_text": chinese,
        "italian_text": italian,
        "chinese_char_span": [prompt.index(chinese), prompt.index(chinese) + len(chinese)],
        "italian_char_span": [prompt.index(italian), prompt.index(italian) + len(italian)],
    }


def all_prompt_records() -> list[dict]:
    return [prompt_record(c, o) for c in CONDITIONS for o in ("forward", "reverse")]


def tokenizer_records(prompt_rec: dict) -> dict:
    output = {}
    for encoder, subdir, max_length in (
        ("clip_l", "tokenizer", 77),
        ("clip_g", "tokenizer_2", 77),
        ("t5", "tokenizer_3", 256),
    ):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / subdir, local_files_only=True)
        encoded = tokenizer(
            prompt_rec["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = [list(pair) for pair in encoded["offset_mapping"]]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        entry = {
            "tokenizer_class": tokenizer.__class__.__name__,
            "max_length_used": max_length,
            "input_ids": ids,
            "tokens": tokens,
            "offset_mapping": offsets,
            "total_non_special_tokens": sum(1 for start, end in offsets if end > start),
        }
        for entity in ("chinese", "italian"):
            char_start, char_end = prompt_rec[f"{entity}_char_span"]
            indices = [
                i for i, (start, end) in enumerate(offsets)
                if end > start and start < char_end and end > char_start
            ]
            entry[f"{entity}_token_indices"] = indices
            entry[f"{entity}_token_count"] = len(indices)
            entry[f"{entity}_tokens"] = [tokens[i] for i in indices]
        output[encoder] = entry
    return output


def prepare_phase1() -> list[dict]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (
        "tokenization",
        "generated_images",
        "attention_mass",
        "value_contribution",
        "layer_timestep_heatmaps",
        "causal_intervention",
    ):
        (RESULTS_DIR / directory).mkdir(exist_ok=True)

    prompts = all_prompt_records()
    with (RESULTS_DIR / "prompts.json").open("w", encoding="utf-8") as handle:
        json.dump(prompts, handle, indent=2, ensure_ascii=False)

    config = {
        "phase": 1,
        "checkpoint": str(MODEL_DIR),
        "checkpoint_class": "StableDiffusion3Pipeline",
        "diffusers_version": "0.39.0",
        "scheduler": "FlowMatchEulerDiscreteScheduler",
        "num_inference_steps": 28,
        "guidance_scale": 4.5,
        "height": 1024,
        "width": 1024,
        "negative_prompt": "",
        "seeds": list(range(20)),
        "paired_design": True,
        "prompt_encoder_mapping": {
            "prompt": "CLIP-L (text_encoder/tokenizer)",
            "prompt_2": "CLIP-G (text_encoder_2/tokenizer_2)",
            "prompt_3": "T5-XXL (text_encoder_3/tokenizer_3)",
        },
        "transformer": {
            "class": "SD3Transformer2DModel",
            "num_layers": 38,
            "num_attention_heads": 38,
            "attention_head_dim": 64,
            "joint_attention_dim": 4096,
        },
        "phase1_scope_note": (
            "Phase 1 establishes SS/LS/SL/LL and prompt-order effects. Attention/value/causal folders "
            "are placeholders required by the final schema and are intentionally not populated in Phase 1."
        ),
    }
    with (RESULTS_DIR / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    summaries = []
    for rec in prompts:
        token_data = tokenizer_records(rec)
        item = {**rec, "encoders": token_data}
        stem = f"{rec['condition']}_{rec['prompt_order']}"
        with (RESULTS_DIR / "tokenization" / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(item, handle, indent=2, ensure_ascii=False)
        summaries.append({
            "condition": rec["condition"],
            "prompt_order": rec["prompt_order"],
            "prompt": rec["prompt"],
            **{
                f"{entity}_token_count_{encoder}": token_data[encoder][f"{entity}_token_count"]
                for encoder in ("clip_l", "clip_g", "t5")
                for entity in ("chinese", "italian")
            },
        })
    with (RESULTS_DIR / "tokenization" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, ensure_ascii=False)
    return summaries


if __name__ == "__main__":
    print(json.dumps(prepare_phase1(), indent=2, ensure_ascii=False))
