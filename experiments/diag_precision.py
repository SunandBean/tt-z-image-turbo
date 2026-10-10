"""Full 8-step parity vs golden under precision variants (matmul / SDPA fidelity and accumulation).
Needs the card (via runner). Uses golden caption features so only the DiT differs. Writes
device-check/diag_precision.json and one PNG per variant for land848."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage.pipeline import Timing, ZImageTurboTT, close_device, open_device  # noqa: E402

OUT = ROOT / "device-check"
CASES = [c for c in os.environ.get("CASES", "").split(",") if c] or ["land848", "sq512"]


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    import ttnn

    dev = open_device()
    report = {"cases": {}}
    try:
        pipe = ZImageTurboTT(dev)
        dit = pipe.dit
        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        H2, H4 = ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
        variants = {
            "default": {"ck_mm": ck(H2, True), "ck_sdpa": ck(H2, False)},
            "sdpa_fp32": {"ck_mm": ck(H2, True), "ck_sdpa": ck(H2, True)},
            "sdpa_hifi4_fp32": {"ck_mm": ck(H2, True), "ck_sdpa": ck(H4, True)},
            "mm_hifi4": {"ck_mm": ck(H4, True), "ck_sdpa": ck(H2, False)},
            "all_hifi4_fp32": {"ck_mm": ck(H4, True), "ck_sdpa": ck(H4, True)},
        }
        for name in CASES:
            gold = torch.load(ROOT / "golden" / name / "tensors.pt")
            case = json.loads((ROOT / "golden" / name / "record.json").read_text())["case"]
            for vname, attrs in variants.items():
                for k, v in attrs.items():
                    setattr(dit, k, v)
                pipe.denoise(gold["cap_feats"], case["width"], case["height"], case["seed"], Timing())  # warm shapes
                tm = Timing()
                lat, _ = pipe.denoise(gold["cap_feats"], case["width"], case["height"], case["seed"], tm)
                rec = {"latents_pcc": pcc(lat, gold["final_latents"].float()), "dit_s": tm.values["dit_s"]}
                if name == "land848":
                    image, _ = pipe.decode(lat)
                    image.save(OUT / f"precision-{name}-{vname}.png")
                report["cases"][f"{name}:{vname}"] = rec
                print(name, vname, json.dumps(rec), flush=True)
                (OUT / "diag_precision.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "diag_precision.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
