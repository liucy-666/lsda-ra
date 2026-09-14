"""One-off demo: generate standalone A/B for P1 (Chinese blue-and-white x Italian maiolica).

Runs at seed 42, 1024x1024, matching the LSDA demo run. Standalone panels only
probe whether the model can render each entity alone (knowledge sufficiency).
"""
import json
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

MODEL = os.environ.get("DEMO_MODEL", "/science/wx/pry/models/stable-diffusion-3.5-large")
OUT = Path(os.environ.get("DEMO_OUT", "/science/wx/pry/MMDIT/experiments/2026_9_12_EXP_1_DEMO_P1/standalone"))
SEED = int(os.environ.get("DEMO_SEED", "42"))
SIZE = 1024
STEPS = 28
GUIDANCE = 4.5

PROMPTS = {
    "A": "Studio photo of a single Chinese blue-and-white porcelain vase on a neutral background, fully visible, centered.",
    "B": "Studio photo of a single Italian maiolica vase on a neutral background, fully visible, centered.",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for tag, prompt in PROMPTS.items():
        path = OUT / f"standalone_{tag}_s{SEED}.png"
        if path.exists():
            print(json.dumps({"skip": str(path)}), flush=True)
            continue
        t0 = time.time()
        generator = torch.Generator("cuda").manual_seed(SEED)
        image = pipe(
            prompt=prompt,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            height=SIZE,
            width=SIZE,
            generator=generator,
        ).images[0]
        image.save(path)
        (OUT / f"standalone_{tag}_s{SEED}.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "seed": SEED,
                    "prompt": prompt,
                    "size": SIZE,
                    "steps": STEPS,
                    "guidance": GUIDANCE,
                    "seconds": round(time.time() - t0, 1),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"done": str(path), "seconds": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
