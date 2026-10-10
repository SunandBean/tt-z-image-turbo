"""CPU: the port's algorithm (ref_algo.RefDiT) on the real weights vs the golden first DiT call.

Proves the folded conditioning, caption-once path and RoPE tables on real weights before a device run."""
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

GOLDEN = Path(os.environ.get("GOLDEN_DIR", ROOT / "golden"))


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "12")))
    algo = ref_algo.RefDiT(host.LazyCheckpoint(snapshot_dir(), "transformer"))
    sched = host.make_scheduler(snapshot_dir())
    t_models, _ = host.model_timesteps(sched)
    results = {}
    for case_dir in sorted(p for p in GOLDEN.iterdir() if (p / "tensors.pt").exists()):
        gold = torch.load(case_dir / "tensors.pt")
        rec = json.loads((case_dir / "record.json").read_text())["case"]
        d0 = gold["dit0"]
        geo = host.Geometry(width=rec["width"], height=rec["height"], cap_len=int(gold["cap_feats"].shape[0]))
        t0 = time.perf_counter()
        with torch.no_grad():
            cap = algo.caption(d0["cap"][0], geo)
            out = algo.step(host.patchify(d0["x"][0].squeeze(1).unsqueeze(0).float(), geo), cap,
                            algo.time.step(t_models[0]), geo)
        ours = host.unpatchify(out.float(), geo)
        ref = d0["out"][0].squeeze(1).unsqueeze(0).float()
        results[case_dir.name] = {"pcc": pcc(ours, ref), "max_abs": float((ours - ref).abs().max()),
                                  "t_golden": float(d0["t"].reshape(-1)[0]), "t_port": t_models[0],
                                  "cpu_s": time.perf_counter() - t0}
        print(json.dumps({case_dir.name: results[case_dir.name]}), flush=True)
    (GOLDEN / "ref_algo_check.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
