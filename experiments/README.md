# Verification scripts

| Script | What it checks |
|---|---|
| `make_golden.py` | Records the diffusers CPU reference the device results are compared against |
| `check_ref_algo.py` | The host maths (scheduler, patchify, latent init) against diffusers |
| `check_ref_full.py` | The full torch reference pipeline against diffusers |
| `check_te_layers.py` | Which text-encoder layer the output is taken from — the 35th, not the demo's 36th |
| `device_check.py` | Each stage on the card, then end to end. Writes `e2e.json` |
| `e2e_check.py` | End-to-end latent and decoded-image PCC against the reference |
| `diag_precision.py` | Precision sweeps behind the bf16 / bfp8 choices |
| `diag_vae.py` | The VAE on the card, including the L1_SMALL exhaustion that forces a program-cache clear per size |
| `diag_geometry.py` | Non-square and non-multiple-of-16 geometry |
| `hires_check.py` | 1920×1080, including the crop from 1088 and the DRAM headroom |
| `stage_service.py` | End-to-end timings through the HTTP service, plus the 422 contract errors |



## Running these outside the tree they were written in

These are the scripts as they were run, inside the private working tree this port was developed in.
They are published as the record behind the numbers on the model card, and most of them need two
edits before they will run from a clone of this repo:

1. **The package name.** Eight of them (`check_ref_algo.py`, `check_ref_full.py`, `device_check.py`,
   `diag_geometry.py`, `diag_precision.py`, `diag_vae.py`, `e2e_check.py`, `hires_check.py`) do
   `from zimage import ...`. `zimage` is this port, published here as **`tt_z_image_turbo`**; the
   module layout is the same, so the import name is the only difference.
2. **The `sys.path` line.** The same eight insert `ROOT.parents[1] / "deploy"`, the private tree's
   package directory. From a clone that is `ROOT.parent`, the repository root.

`make_golden.py` and `check_te_layers.py` run as published. `stage_service.py` drives the private
deployment's own HTTP service and is a record rather than something to run. The device scripts need
a p100a either way.
