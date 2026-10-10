"""Where does the TT DiT drift from ref_algo, and which setting causes it? Needs the card (via runner).

For a few geometries (with / without image pad tokens, different SDPA chunk sizes) one DiT step is run on
the device with per-block taps and compared with ref_algo on CPU (same inputs, real weights). Variants:
default, SDPA chunk forced to 32, matmul HiFi4. Writes device-check/diag_geometry.json."""
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage import host, ref_algo  # noqa: E402
from zimage.config import snapshot_dir  # noqa: E402
from zimage.pipeline import close_device, open_device  # noqa: E402

OUT = ROOT / "device-check"
OUT.mkdir(exist_ok=True)
GEOS = [(512, 512), (832, 640), (848, 624)]
T_INDEX = int(os.environ.get("T_INDEX", "4"))  # a mid-schedule step (t_model 0.25) exercises real conditioning


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    import ttnn
    from zimage.tt_dit import ZImageDiT

    torch.set_num_threads(16)
    snap = snapshot_dir()
    cap_feats = torch.load(ROOT / "golden/land848/tensors.pt")["cap_feats"]
    ckpt = host.LazyCheckpoint(snap, "transformer")
    ref = ref_algo.RefDiT(ckpt)
    t_models, _ = host.model_timesteps(host.make_scheduler(snap))
    t = t_models[T_INDEX]
    cond = ref.time.step(t)
    report = {"t_model": t, "cases": {}}
    ref_path = OUT / f"diag_refs_t{T_INDEX}.pt"
    refs = torch.load(ref_path, weights_only=False) if ref_path.exists() else {}
    for w, h in GEOS:
        if (w, h) in refs:
            continue
        geo = host.Geometry(width=w, height=h, cap_len=int(cap_feats.shape[0]))
        g = torch.Generator().manual_seed(0)
        lat = torch.randn(1, 16, geo.lat_h, geo.lat_w, generator=g)
        taps = []
        t0 = time.perf_counter()
        with torch.no_grad():
            out = ref.step(host.patchify(lat, geo), ref.caption(cap_feats, geo), cond, geo, taps=taps)
        refs[(w, h)] = (geo, lat, taps, out.float())
        print(f"ref {w}x{h} n_img={geo.n_img} pad={geo.n_img_pad - geo.n_img} S={geo.n_joint} "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
    torch.save(refs, ref_path)
    if os.environ.get("REF_ONLY") == "1":
        return

    dev = open_device()
    try:
        dit = ZImageDiT(dev, ckpt)
        dit.load_schedule([t])
        arch = dev.arch()
        variants = {
            "default": {},
            "sdpa32": {"sdpa_chunks": (32,)},
            "mm_hifi4": {"ck_mm": ttnn.init_device_compute_kernel_config(
                arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
                packer_l1_acc=False)},
        }
        base = {"sdpa_chunks": dit.sdpa_chunks, "ck_mm": dit.ck_mm}
        for (w, h), (geo, lat, rtaps, rout) in refs.items():
            prep = dit.prepare(geo)
            cap = dit.caption(cap_feats, prep)
            for name, attrs in variants.items():
                for k, v in {**base, **attrs}.items():
                    setattr(dit, k, v)
                taps = []
                t0 = time.perf_counter()
                out = dit.step(host.patchify(lat, geo), cap, t, prep, taps=taps)
                per_block = [pcc(a, b[: a.shape[0]]) for a, b in zip(taps, rtaps)]
                key = f"{w}x{h}:{name}"
                report["cases"][key] = {
                    "pad_tokens": geo.n_img_pad - geo.n_img, "S": geo.n_joint, "step_s": time.perf_counter() - t0,
                    "out_pcc": pcc(out[: geo.n_img], rout[: geo.n_img]),
                    "embed_pcc": per_block[0], "after_refiners": per_block[2],
                    "blocks_min": min(per_block), "per_block": per_block,
                    # image rows only (pads excluded) after the last block
                    "last_img_rows_pcc": pcc(taps[-1][: geo.n_img], rtaps[-1][: geo.n_img]),
                }
                print(key, json.dumps({k: v for k, v in report["cases"][key].items() if k != "per_block"}), flush=True)
                (OUT / "diag_geometry.json").write_text(json.dumps(report, indent=2))
            ttnn.deallocate(cap)
            dit.release(prep)
    finally:
        (OUT / "diag_geometry.json").write_text(json.dumps(report, indent=2))
        close_device(dev)


if __name__ == "__main__":
    main()
