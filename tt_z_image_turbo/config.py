# SPDX-License-Identifier: Apache-2.0
"""Static configuration of Tongyi-MAI/Z-Image-Turbo @ f332072 (transformer/text_encoder/scheduler configs)."""
from __future__ import annotations

import os
from dataclasses import dataclass

HF_REPO = "Tongyi-MAI/Z-Image-Turbo"
HF_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
TILE = 32
SEQ_MULTI_OF = 32  # diffusers transformer_z_image: captions and image patches are padded to this


@dataclass(frozen=True)
class DiTConfig:
    dim: int = 3840
    heads: int = 30
    head_dim: int = 128
    mlp_hidden: int = 10240  # int(dim / 3 * 8)
    n_layers: int = 30
    n_refiner_layers: int = 2
    patch: int = 2
    in_channels: int = 16
    cap_feat_dim: int = 2560
    axes_dims: tuple = (32, 48, 48)
    axes_lens: tuple = (1536, 512, 512)
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    norm_eps: float = 1e-5  # RMSNorm of blocks, qk-norm and cap_embedder
    final_eps: float = 1e-6  # final LayerNorm (no affine)
    adaln_dim: int = 256  # min(dim, ADALN_EMBED_DIM)

    @property
    def patch_dim(self) -> int:
        return self.patch * self.patch * self.in_channels


@dataclass(frozen=True)
class TextEncoderConfig:
    num_layers: int = 35  # hidden_states[-2] == residual stream after 35 of 36 layers (check_te_layers.py)
    hidden: int = 2560
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    intermediate: int = 9728
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_tokens: int = 512  # ZImagePipeline max_sequence_length


@dataclass(frozen=True)
class SchedulerConfig:
    num_train_timesteps: int = 1000
    shift: float = 3.0  # use_dynamic_shifting false: resolution-independent schedule
    steps: int = 9  # model card: 9 scheduler steps = 8 DiT forwards (the last sigma step is 0)


DIT = DiTConfig()
TE = TextEncoderConfig()
SCHED = SchedulerConfig()
VAE_SCALING = 0.3611
VAE_SHIFT = 0.1159
VAE_SCALE_FACTOR = 8


def snapshot_dir() -> str:
    """The local Z-Image-Turbo snapshot, downloaded into the HF cache if it is not there yet.

    Z_IMAGE_SNAPSHOT overrides it with a directory you manage yourself.
    """
    env = os.environ.get("Z_IMAGE_SNAPSHOT")
    if env:
        return env
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    )
    d = os.path.join(cache, "models--" + HF_REPO.replace("/", "--"), "snapshots", HF_REVISION)
    if os.path.isdir(d):
        return d
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=HF_REPO, revision=HF_REVISION)
