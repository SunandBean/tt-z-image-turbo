"""P100a parity + timing check of the single-chip Z-Image-Turbo port against the CPU goldens.

Needs exclusive use of the card (the photo model container must be stopped). Writes device-check/report.json:
  * text encoder: caption features vs golden hidden_states[-2]
  * DiT: first step output vs golden (same inputs), full 8-step latents vs golden, decoded image
  * timings (load, TE, caption, per step, VAE) and DRAM after load / after each case
Stages are recorded as they finish so a crash still leaves the partial report."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage import host  # noqa: E402
from zimage.pipeline import Timing, ZImageTurboTT, close_device, dram_stats, open_device  # noqa: E402

GOLDEN = Path(os.environ.get("GOLDEN_DIR", ROOT / "golden"))
OUT = Path(os.environ.get("CHECK_OUT", ROOT / "device-check"))
OUT.mkdir(parents=True, exist_ok=True)
report = {"status": "starting", "cases": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    cases = sorted(p.name for p in GOLDEN.iterdir() if (p / "tensors.pt").exists())
    only = set(filter(None, os.environ.get("CASES", "").split(",")))
    cases = [c for c in cases if not only or c in only]
    dev = open_device()
    try:
        t0 = time.perf_counter()
        pipe = ZImageTurboTT(dev)
        report.update(status="loaded", load_s=time.perf_counter() - t0, dram_after_load=dram_stats(dev),
                      t_models=pipe.t_models)
        save()
        for name in cases:
            rec = json.loads((GOLDEN / name / "record.json").read_text())
            gold = torch.load(GOLDEN / name / "tensors.pt")
            case = rec["case"]
            r = report["cases"][name] = {"case": case}
            timing = Timing()

            t0 = time.perf_counter()
            cap = pipe.encode(case["prompt"])
            timing.mark("text_encoder_s", t0)
            r["cap_tokens"] = [int(cap.shape[0]), int(gold["cap_feats"].shape[0])]
            r["te_pcc"] = pcc(cap.float(), gold["cap_feats"].float())
            r["te_max_abs"] = float((cap.float() - gold["cap_feats"].float()).abs().max())
            save()

            # DiT step 0 on the golden's exact inputs (golden caption features and noise)
            geo = host.Geometry(width=case["width"], height=case["height"], cap_len=int(gold["cap_feats"].shape[0]))
            d0 = gold["dit0"]
            prep = pipe.prepared(geo)
            capd = pipe.dit.caption(gold["cap_feats"], prep)
            t_model = float(d0["t"].reshape(-1)[0])
            lat0 = d0["x"][0].squeeze(1).unsqueeze(0).float()  # [16,1,H,W] -> [1,16,H,W]
            t0 = time.perf_counter()
            out0 = pipe.dit.step(host.patchify(lat0, geo), capd, pipe.t_models[0], prep)
            r["first_step_s"] = time.perf_counter() - t0
            import ttnn
            ttnn.deallocate(capd)
            ref0 = d0["out"][0].squeeze(1).unsqueeze(0).float()
            r["t_model_golden_vs_port"] = [t_model, pipe.t_models[0]]
            r["dit_step0_pcc"] = pcc(host.unpatchify(out0, geo), ref0)
            save()

            # full generation from golden caption features, then from our own TE
            for label, feats in (("golden_te", gold["cap_feats"]), ("tt_te", cap)):
                tm = Timing()
                lat, _ = pipe.denoise(feats, case["width"], case["height"], case["seed"], tm)
                t0 = time.perf_counter()
                image, decoded = pipe.decode(lat)
                tm.mark("vae_decode_s", t0)
                image.save(OUT / f"{name}-{label}.png")
                r[label] = {"latents_pcc": pcc(lat, gold["final_latents"].float()),
                            "decoded_pcc": pcc(decoded.float(), gold["decoded"]),
                            "timing_s": tm.values}
                save()
            r["dram_after_case"] = dram_stats(dev)
            save()
        report["status"] = "ok"
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        save()
        close_device(dev)


if __name__ == "__main__":
    main()
