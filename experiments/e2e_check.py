"""End-to-end Z-Image-Turbo on the P100a (TT text encoder + DiT + VAE) over a sequence of sizes and prompts
like a real queue: repeated sizes, size changes, different prompt lengths. Needs the card (via runner).
Writes device-check/e2e.json and e2e-*.png; golden cases also get latents / image PCC."""
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage.pipeline import ZImageTurboTT, close_device, dram_stats, open_device  # noqa: E402

OUT = ROOT / "device-check"
LONG = ("An elderly fisherman mending a bright orange net on a wooden pier at sunrise, seagulls overhead, "
        "fog over the harbor, detailed weathered hands, cinematic 35mm photograph, shallow depth of field")
SEQ = [
    ("sq512", None, 512, 512, None),
    ("sq512-repeat", "sq512", 512, 512, None),
    ("croissant512", "A croissant beside a cup of coffee, food photograph", 512, 512, 3),
    ("land848", None, 848, 624, None),
    ("odd528x784", "A lighthouse on a rocky cliff during a storm, oil painting", 528, 784, 11),
    ("sq1024-long", LONG, 1024, 1024, 5),
    ("land1024x768", "Neon-lit Tokyo street at night in the rain, reflections on wet asphalt", 1024, 768, 21),
    ("port576x1024", "Portrait of a red fox in fresh snow, wildlife photography", 576, 1024, 9),
    ("sq512-back", "sq512", 512, 512, None),
]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    report = {"runs": []}
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = ZImageTurboTT(dev)
        report.update(load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev))
        for name, prompt_or_case, w, h, seed in SEQ:
            rec = {"name": name, "width": w, "height": h}
            gold = None
            if prompt_or_case is None or prompt_or_case in ("sq512", "land848"):
                case_name = prompt_or_case or name
                case = json.loads((ROOT / "golden" / case_name / "record.json").read_text())["case"]
                gold = torch.load(ROOT / "golden" / case_name / "tensors.pt")
                prompt, seed = case["prompt"], case["seed"]
            else:
                prompt = prompt_or_case
            rec.update(prompt=prompt, seed=seed)
            try:
                from zimage.pipeline import Timing

                timing = Timing()
                started = time.perf_counter()
                t1 = time.perf_counter()
                cap = pipe.encode(prompt)
                timing.mark("text_encoder_s", t1)
                lat, _ = pipe.denoise(cap, w, h, seed, timing)
                t1 = time.perf_counter()
                image, decoded = pipe.decode(lat)
                timing.mark("vae_decode_s", t1)
                rec["total_s"] = time.perf_counter() - started
                rec["timing_s"] = timing.values
                rec["cap_tokens"] = int(cap.shape[0])
                arr = np.asarray(image)
                rec["size"] = list(image.size)
                rec["pixel_std"] = float(arr.std())
                if gold is not None:
                    rec["latents_pcc"] = pcc(lat, gold["final_latents"].float())
                    rec["decoded_pcc"] = pcc(decoded.float(), gold["decoded"].float())
                image.save(OUT / f"e2e-{name}.png")
                rec["dram"] = dram_stats(dev)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                rec["traceback"] = traceback.format_exc()[-3000:]
            report["runs"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("traceback", "dram", "prompt")}), flush=True)
            (OUT / "e2e.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "e2e.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
