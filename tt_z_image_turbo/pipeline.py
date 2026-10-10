# SPDX-License-Identifier: Apache-2.0
"""Z-Image-Turbo text-to-image on one P100a: TT text encoder + TT DiT, host scheduler, CPU VAE decode."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from . import host
from .config import DIT, VAE_SCALING, VAE_SHIFT, snapshot_dir


def open_device(trace_region_size: int = 0, l1_small_size: int = 98304):
    import ttnn

    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
        trace_region_size=trace_region_size,
        l1_small_size=l1_small_size,
    )


def close_device(dev):
    import ttnn

    ttnn.close_mesh_device(dev)


def dram_stats(dev) -> Dict[str, int]:
    import ttnn

    ttnn.synchronize_device(dev)
    v = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
    banks = int(v.num_banks)
    return {"total_bytes": int(v.total_bytes_per_bank) * banks,
            "allocated_bytes": int(v.total_bytes_allocated_per_bank) * banks,
            "free_bytes": int(v.total_bytes_free_per_bank) * banks,
            "largest_free_bytes_per_bank": int(v.largest_contiguous_bytes_free_per_bank)}


@dataclass
class Timing:
    values: Dict[str, float] = field(default_factory=dict)

    def mark(self, key: str, started: float):
        self.values[key] = self.values.get(key, 0.0) + time.perf_counter() - started


class ZImageTurboTT:
    def __init__(self, dev, snapshot: Optional[str] = None, dit_prec=None, te_prec=None, tt_vae: bool = True):
        from transformers import AutoTokenizer
        import ttnn

        from .tt_dit import ZImageDiT
        from .tt_text_encoder import ZImageTextEncoder

        self.dev = dev
        self.snapshot = snapshot or snapshot_dir()
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.snapshot, "tokenizer"))
        self.te = ZImageTextEncoder(dev, host.LazyCheckpoint(self.snapshot, "text_encoder"), te_prec)
        self.dit = ZImageDiT(dev, host.LazyCheckpoint(self.snapshot, "transformer"), dit_prec)
        self.scheduler_template = host.make_scheduler(self.snapshot)
        self.t_models, self.active_steps = host.model_timesteps(self.scheduler_template)
        self.dit.load_schedule(self.t_models)
        self.tt_vae = None
        if tt_vae:
            from .tt_vae import ZImageVAE

            self.tt_vae = ZImageVAE(dev, self.snapshot)
        ttnn.synchronize_device(dev)
        self.load_s = time.perf_counter() - t0
        self._vae = None
        self._prep = None

    @property
    def vae(self):
        if self._vae is None:
            from diffusers import AutoencoderKL

            self._vae = AutoencoderKL.from_pretrained(self.snapshot, subfolder="vae", torch_dtype=torch.bfloat16).eval()
        return self._vae

    def prepared(self, geo: host.Geometry):
        if self._prep is None or self._prep.geo != geo:
            if self._prep is not None:
                self.dit.release(self._prep)
                if (self._prep.geo.width, self._prep.geo.height) != (geo.width, geo.height):
                    # conv2d keeps per-shape sliding-window configs in L1_SMALL inside cached programs; they
                    # would exhaust it after a few sizes (diag_vae.py), so a new size starts a clean cache
                    self.dev.clear_program_cache()
            self._prep = self.dit.prepare(geo)
        return self._prep

    def encode(self, prompt: str) -> torch.Tensor:
        return self.te.encode(host.tokenize(self.tokenizer, prompt))

    def denoise(self, cap_feats: torch.Tensor, width: int, height: int, seed: int, timing: Timing,
                return_first: bool = False):
        import ttnn

        geo = host.Geometry(width=width, height=height, cap_len=int(cap_feats.shape[0]))
        t0 = time.perf_counter()
        prep = self.prepared(geo)
        cap = self.dit.caption(cap_feats, prep)
        timing.mark("caption_s", t0)
        latents = host.init_latents(geo, seed)
        sched = self._fresh_scheduler()
        first = None
        for i, t_model in zip(self.active_steps, self.t_models):
            t0 = time.perf_counter()
            out = self.dit.step(host.patchify(latents, geo), cap, t_model, prep)
            timing.mark("dit_s", t0)
            if return_first and first is None:
                first = host.unpatchify(out, geo)
            noise_pred = -host.unpatchify(out, geo)
            latents = sched.step(noise_pred.float(), sched.timesteps[i], latents, return_dict=False)[0]
        ttnn.deallocate(cap)
        return latents, first

    def _fresh_scheduler(self):
        import copy

        return copy.deepcopy(self.scheduler_template)

    def decode(self, latents: torch.Tensor):
        with torch.no_grad():
            if self.tt_vae is not None:
                image = self.tt_vae.decode(latents)
            else:
                scaled = latents.to(torch.bfloat16) / VAE_SCALING + VAE_SHIFT
                image = self.vae.decode(scaled, return_dict=False)[0]
        from diffusers.image_processor import VaeImageProcessor

        return VaeImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")[0], image

    def generate(self, prompt: str, width: int, height: int, seed: int):
        timing = Timing()
        t0 = time.perf_counter()
        cap = self.encode(prompt)
        timing.mark("text_encoder_s", t0)
        latents, _ = self.denoise(cap, width, height, seed, timing)
        t0 = time.perf_counter()
        image, _ = self.decode(latents)
        timing.mark("vae_decode_s", t0)
        return image, timing
