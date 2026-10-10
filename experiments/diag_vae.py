"""TT VAE decoder vs the CPU diffusers decoder: parity on the golden latents and timing on several sizes.
Needs the card (via runner). Writes device-check/diag_vae.json and TT-decoded PNGs of the golden cases."""
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage.config import snapshot_dir  # noqa: E402
from zimage.pipeline import close_device, dram_stats, open_device  # noqa: E402

OUT = ROOT / "device-check"
EXTRA = [(1024, 1024), (1024, 768), (528, 784)]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    from diffusers.image_processor import VaeImageProcessor

    from zimage.tt_vae import ZImageVAE

    proc = VaeImageProcessor(vae_scale_factor=16)
    report = {"cases": {}}
    dev = open_device()
    try:
        t0 = time.perf_counter()
        vae = ZImageVAE(dev, snapshot_dir())
        report["load_s"] = time.perf_counter() - t0
        report["dram_after_load"] = dram_stats(dev)
        items = []
        for name in ("sq512", "land848"):
            gold = torch.load(ROOT / "golden" / name / "tensors.pt")
            items.append((name, gold["final_latents"].float(), gold["decoded"].float()))
        g = torch.Generator().manual_seed(0)
        for w, h in EXTRA:
            items.append((f"rand{w}x{h}", torch.randn(1, 16, h // 8, w // 8, generator=g), None))
        for name, lat, ref in items:
            rec = {}
            try:
                for run in ("first", "second"):
                    t0 = time.perf_counter()
                    out = vae.decode(lat)
                    rec[f"{run}_s"] = time.perf_counter() - t0
                rec["shape"] = list(out.shape)
                if ref is not None:
                    rec["pcc_vs_cpu"] = pcc(out, ref)
                    rec["max_abs"] = float((out - ref).abs().max())
                    proc.postprocess(out, output_type="pil")[0].save(OUT / f"vae-{name}.png")
                rec["dram"] = dram_stats(dev)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rec["traceback"] = traceback.format_exc()[-3000:]
            report["cases"][name] = rec
            print(name, json.dumps({k: v for k, v in rec.items() if k not in ("traceback", "dram")}), flush=True)
            (OUT / "diag_vae.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "diag_vae.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
