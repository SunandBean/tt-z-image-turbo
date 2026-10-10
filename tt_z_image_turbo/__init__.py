# SPDX-License-Identifier: Apache-2.0
"""Z-Image-Turbo text-to-image on one Tenstorrent Blackhole p100a.

The Qwen3-4B text encoder, the DiT and the VAE all run on the card in TTNN; only the
scheduler and the image post-processing stay on the host.
"""
from .pipeline import Timing, ZImageTurboTT, close_device, dram_stats, open_device

__all__ = ["ZImageTurboTT", "Timing", "open_device", "close_device", "dram_stats"]
__version__ = "0.1.0"
