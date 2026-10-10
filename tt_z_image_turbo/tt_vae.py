# SPDX-License-Identifier: Apache-2.0
"""Z-Image VAE decoder (the FLUX.1 16-channel AutoencoderKL) on one Blackhole.

Uses tt-metal's tt_dit SD3.5/FLUX VAE decoder (models/tt_dit/models/vae/vae_sd35.py, Apache-2.0), vendored
from tt-metal 01d6e7b into vendor_tt_dit/ because the runtime image ships only part of tt_dit. On a 1x1 mesh
every all-gather in it is a no-op; vendor_tt_dit/layers/conv2d.py adds width slicing for non-square sizes."""
from __future__ import annotations

import torch
import ttnn

from .config import VAE_SCALING, VAE_SHIFT
from .vendor_tt_dit.models.vae.vae_sd35 import VAEDecoder
from .vendor_tt_dit.parallel.config import VAEParallelConfig
from .vendor_tt_dit.parallel.manager import CCLManager
from .vendor_tt_dit.utils import tensor as tt_tensor


class ZImageVAE:
    def __init__(self, dev, snapshot: str):
        from diffusers import AutoencoderKL

        torch_vae = AutoencoderKL.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32).eval()
        self.dev = dev
        self.ccl = CCLManager(dev, num_links=1, topology=ttnn.Topology.Linear)
        self.decoder = VAEDecoder.from_torch(torch_vae.decoder, mesh_device=dev,
                                             parallel_config=VAEParallelConfig.from_tuple((1, 1)), ccl_manager=self.ccl)
        del torch_vae

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """[1, 16, h, w] float latents -> [1, 3, 8h, 8w] float image in [-1, 1] (before postprocess)."""
        z = (latents.float() / VAE_SCALING + VAE_SHIFT).permute(0, 2, 3, 1)  # NHWC like VAEDecoderAdapter
        tt_in = tt_tensor.from_torch(z, device=self.dev)
        tt_out = self.decoder.forward(tt_in)
        out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2).float()
        ttnn.deallocate(tt_out)
        ttnn.deallocate(tt_in)
        return out
