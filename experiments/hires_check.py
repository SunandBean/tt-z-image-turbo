"""Can Z-Image-Turbo on the P100a make 1920x1080? Generates at 1920x1088 (16 px grid) and center-crops 8 rows.
Needs the card (via runner). Writes device-check/hires.json and hires-*.png (cropped) images."""
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage.pipeline import ZImageTurboTT, close_device, dram_stats, open_device  # noqa: E402

OUT = ROOT / "device-check"
CASES = [
    ("city", "Aerial view of a coastal city at golden hour, cinematic wide shot, highly detailed", 1920, 1088, 1),
    ("city-warm", "Aerial view of a coastal city at golden hour, cinematic wide shot, highly detailed", 1920, 1088, 1),
    ("forest", "A misty pine forest with a wooden cabin and a winding river, landscape photograph", 1920, 1088, 2),
    ("studio", "Two friends laughing at a cafe table by a large window, natural light, 35mm photo", 1920, 1088, 3),
    ("anime", "A girl riding a bicycle along a seaside road under summer clouds, anime key visual", 1920, 1088, 4),
    ("portrait", "A tall lighthouse on a cliff under a starry night sky, vertical composition", 1088, 1920, 5),
]


def crop_to_1080(image):
    w, h = image.size
    if (w, h) == (1920, 1088):
        return image.crop((0, 4, 1920, 1084))
    if (w, h) == (1088, 1920):
        return image.crop((4, 0, 1084, 1920))
    return image


def main():
    report = {"runs": []}
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = ZImageTurboTT(dev)
        report.update(load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev))
        for name, prompt, w, h, seed in CASES:
            rec = {"name": name, "width": w, "height": h, "seed": seed, "prompt": prompt}
            try:
                started = time.perf_counter()
                image, timing = pipe.generate(prompt, w, h, seed)
                rec["total_s"] = time.perf_counter() - started
                rec["timing_s"] = timing.values
                rec["dram"] = dram_stats(dev)
                out = crop_to_1080(image)
                rec["saved_size"] = list(out.size)
                out.save(OUT / f"hires-{name}.png")
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
                rec["traceback"] = traceback.format_exc()[-3000:]
            report["runs"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("traceback", "prompt", "dram")}), flush=True)
            (OUT / "hires.json").write_text(json.dumps(report, indent=2))
    finally:
        (OUT / "hires.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
