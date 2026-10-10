"""CPU golden references for the single-P100a Z-Image-Turbo port. No accelerator access.

For each case this runs the pinned diffusers ZImagePipeline in bf16 and saves the
tensors a TT port must reproduce: caption features (hidden_states[-2], valid tokens only),
the first DiT call's inputs/output, final latents, the decoded image, and CPU timings."""
import json
import os
from pathlib import Path
import time

import torch
from diffusers import ZImagePipeline

SNAPSHOT = Path(os.environ["Z_IMAGE_SNAPSHOT"])
OUT = Path(os.environ.get("GOLDEN_OUT", Path(__file__).resolve().parent / "golden"))
STEPS = 9  # Turbo model card: 9 scheduler steps = 8 DiT forwards
CASES = [
    {"name": "sq512", "prompt": "a misty mountain lake at dawn, watercolor painting", "width": 512, "height": 512, "seed": 42},
    {"name": "land848", "prompt": "A cozy reading nook with a cat sleeping on a knitted blanket, warm afternoon light",
     "width": 848, "height": 624, "seed": 7},
]


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    pipe = ZImagePipeline.from_pretrained(SNAPSHOT, torch_dtype=torch.bfloat16)
    selected = set(filter(None, os.environ.get("CASES", "").split(",")))
    for case in CASES:
        if selected and case["name"] not in selected:
            continue
        out = OUT / case["name"]
        out.mkdir(parents=True, exist_ok=True)
        record = {"case": case, "steps": STEPS, "timings_s": {}}

        t0 = time.perf_counter()
        with torch.no_grad():
            cap = pipe._encode_prompt(case["prompt"], device="cpu")[0]
        record["timings_s"]["text_encoder"] = time.perf_counter() - t0
        record["cap_tokens"] = int(cap.shape[0])

        calls = []

        def hook(module, args, kwargs, output):
            if not calls:  # pipeline calls transformer(x_list, t, cap_list, return_dict=False)
                calls.append({"x": [a.detach().clone() for a in args[0]], "t": args[1].detach().clone(),
                              "cap": [c.detach().clone() for c in args[2]],
                              "out": [o.detach().clone() for o in output[0]]})
            record["timings_s"].setdefault("dit_calls", []).append(time.perf_counter() - hook.started)
            hook.started = time.perf_counter()

        handle = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        generator = torch.Generator("cpu").manual_seed(case["seed"])
        hook.started = time.perf_counter()
        t0 = time.perf_counter()
        latents = pipe(prompt_embeds=[cap], height=case["height"], width=case["width"], num_inference_steps=STEPS,
                       guidance_scale=0.0, generator=generator, output_type="latent").images
        record["timings_s"]["denoise"] = time.perf_counter() - t0
        handle.remove()

        t0 = time.perf_counter()
        with torch.no_grad():
            scaled = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            decoded = pipe.vae.decode(scaled, return_dict=False)[0]
        record["timings_s"]["vae_decode"] = time.perf_counter() - t0
        image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
        image.save(out / "image.png")

        torch.save({"cap_feats": cap, "dit0": calls[0], "final_latents": latents, "decoded": decoded.float()},
                   out / "tensors.pt")
        record["dit_forwards"] = len(record["timings_s"].get("dit_calls", []))
        (out / "record.json").write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
