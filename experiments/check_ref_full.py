"""CPU baseline: how far does a mathematically identical but differently rounded implementation (ref_algo)
drift from the diffusers golden over the full 8 steps? Sets the realistic parity bar for the device."""
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "deploy"))
from zimage import host, ref_algo  # noqa: E402
from zimage.config import snapshot_dir  # noqa: E402

GOLDEN = ROOT / "golden"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    snap = snapshot_dir()
    algo = ref_algo.RefDiT(host.LazyCheckpoint(snap, "transformer"))
    sched0 = host.make_scheduler(snap)
    t_models, active = host.model_timesteps(sched0)
    conds = [algo.time.step(t) for t in t_models]
    out = {}
    for name in os.environ.get("CASES", "land848,sq512").split(","):
        gold = torch.load(GOLDEN / name / "tensors.pt")
        case = json.loads((GOLDEN / name / "record.json").read_text())["case"]
        geo = host.Geometry(width=case["width"], height=case["height"], cap_len=int(gold["cap_feats"].shape[0]))
        import copy
        sched = copy.deepcopy(sched0)
        lat = host.init_latents(geo, case["seed"])
        with torch.no_grad():
            cap = algo.caption(gold["cap_feats"], geo)
            for i, c in zip(active, conds):
                v = host.unpatchify(algo.step(host.patchify(lat, geo), cap, c, geo).float(), geo)
                lat = sched.step(-v, sched.timesteps[i], lat, return_dict=False)[0]
        out[name] = {"latents_pcc_vs_golden": pcc(lat, gold["final_latents"].float())}
        print(name, out[name], flush=True)
    (GOLDEN / "ref_full_check.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
