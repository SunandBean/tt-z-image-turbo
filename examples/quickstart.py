# SPDX-License-Identifier: Apache-2.0
"""Minimal text-to-image run on one p100a.

    python examples/quickstart.py "a misty mountain lake at dawn, watercolor painting"
"""
import sys

from tt_z_image_turbo import ZImageTurboTT, close_device, dram_stats, open_device

prompt = sys.argv[1] if len(sys.argv) > 1 else "a misty mountain lake at dawn, watercolor painting"

dev = open_device()
try:
    model = ZImageTurboTT(dev)   # downloads the snapshot into the HF cache on first use
    print(f"loaded in {model.load_s:.1f} s, device DRAM {dram_stats(dev)['allocated_bytes'] / 2**30:.1f} GiB")

    image, timing = model.generate(prompt, 1024, 1024, seed=42)
    image.save("output.png")
    print({k: round(v, 2) for k, v in timing.values.items()})
finally:
    close_device(dev)
