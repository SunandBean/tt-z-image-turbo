# SPDX-License-Identifier: Apache-2.0
"""Host-side (CPU) parts of the single-P100a Z-Image-Turbo port.

Everything here mirrors diffusers 0.37.1 ZImagePipeline / ZImageTransformer2DModel for batch 1 and is
checked against it in tests/test_host.py:
  * caption / image token counts padded to SEQ_MULTI_OF, patchify / unpatchify
  * RoPE positions and cos/sin tables in the adjacent-pair layout of rotary_embedding_llama
  * timestep conditioning precomputed for the fixed 9-step schedule, folded into RMSNorm weights
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from safetensors import safe_open

from .config import DIT, SCHED, SEQ_MULTI_OF, TE, TILE, DiTConfig


def round_up(n: int, m: int = SEQ_MULTI_OF) -> int:
    return (n + m - 1) // m * m


# ----------------------------------------------------------------------------- checkpoint access
class LazyCheckpoint:
    """safetensors shards of one sub-folder, opened lazily; get() returns an owned tensor."""

    def __init__(self, root: str, subfolder: str):
        self.root = os.path.join(root, subfolder)
        idx = [f for f in os.listdir(self.root) if f.endswith(".safetensors.index.json")]
        if idx:
            weight_map = json.load(open(os.path.join(self.root, idx[0])))["weight_map"]
            self.key_to_file = {k: os.path.join(self.root, v) for k, v in weight_map.items()}
        else:
            (single,) = [f for f in os.listdir(self.root) if f.endswith(".safetensors")]
            path = os.path.join(self.root, single)
            with safe_open(path, framework="pt") as f:
                self.key_to_file = {k: path for k in f.keys()}
        self._handles: Dict[str, object] = {}

    def _handle(self, path):
        if path not in self._handles:
            self._handles[path] = safe_open(path, framework="pt", device="cpu")
        return self._handles[path]

    def keys(self):
        return self.key_to_file.keys()

    def get(self, key: str, dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
        t = self._handle(self.key_to_file[key]).get_tensor(key)
        return t.to(dtype) if dtype is not None and t.dtype != dtype else t.clone()

    def get_rows(self, key: str, rows: torch.Tensor) -> torch.Tensor:
        sl = self._handle(self.key_to_file[key]).get_slice(key)
        return torch.stack([sl[i : i + 1][0] for i in rows.reshape(-1).tolist()], dim=0)


# ----------------------------------------------------------------------------- weight layout helpers
def linear_to_mm(w: torch.Tensor) -> torch.Tensor:
    """nn.Linear weight [out, in] -> matmul weight [in, out]."""
    return w.t().contiguous()


def fuse_qkv(wq, wk, wv) -> torch.Tensor:
    return torch.cat([linear_to_mm(wq), linear_to_mm(wk), linear_to_mm(wv)], dim=1).contiguous()


def swiglu_interleave(gate_out_in: torch.Tensor, up_out_in: torch.Tensor, tile: int = TILE) -> torch.Tensor:
    """[K, 2N] weight for minimal_matmul(fuse_swiglu=True): column tile 2p = gate tile p, 2p+1 = up tile p."""
    g, u = linear_to_mm(gate_out_in), linear_to_mm(up_out_in)
    K, N = g.shape
    assert N % tile == 0, N
    return torch.stack([g.view(K, N // tile, tile), u.view(K, N // tile, tile)], dim=2).reshape(K, 2 * N).contiguous()


def interleave_pairs_permutation(head_dim: int) -> torch.Tensor:
    """new[j] = old[p[j]]: llama rotate_half pairs (i, i + D/2) -> adjacent pairs (2i, 2i + 1)."""
    half = head_dim // 2
    p = torch.empty(head_dim, dtype=torch.long)
    p[0::2] = torch.arange(half)
    p[1::2] = torch.arange(half) + half
    return p


def permute_heads_rows(w_out_in: torch.Tensor, n_heads: int, head_dim: int, perm: torch.Tensor) -> torch.Tensor:
    out, inp = w_out_in.shape
    assert out == n_heads * head_dim
    return w_out_in.view(n_heads, head_dim, inp)[:, perm, :].reshape(out, inp).contiguous()


def rot_transformation_mat(tile: int = TILE) -> torch.Tensor:
    """[1, 1, 32, 32] T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent-pair rotation)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[-2] == rows:
        return t
    out = torch.zeros(*t.shape[:-2], rows, t.shape[-1], dtype=t.dtype)
    out[..., : t.shape[-2], :] = t
    return out


# ----------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class Geometry:
    """Token layout of one generation: image patches (padded) followed by caption tokens (padded)."""

    width: int
    height: int
    cap_len: int  # valid caption tokens L (<= TE.max_tokens)

    @property
    def lat_h(self) -> int:
        return 2 * (self.height // 16)

    @property
    def lat_w(self) -> int:
        return 2 * (self.width // 16)

    @property
    def grid_h(self) -> int:
        return self.lat_h // DIT.patch

    @property
    def grid_w(self) -> int:
        return self.lat_w // DIT.patch

    @property
    def n_img(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def n_img_pad(self) -> int:
        return round_up(self.n_img)

    @property
    def n_cap_pad(self) -> int:
        return round_up(self.cap_len)

    @property
    def n_joint(self) -> int:
        return self.n_img_pad + self.n_cap_pad


def patchify(latents: torch.Tensor, geo: Geometry) -> torch.Tensor:
    """[1, 16, H, W] -> [n_img_pad, 64] (diffusers _patchify_image; pad rows are zero, replaced on device)."""
    c, p = DIT.in_channels, DIT.patch
    x = latents.reshape(c, 1, 1, geo.grid_h, p, geo.grid_w, p)
    x = x.permute(1, 3, 5, 2, 4, 6, 0).reshape(geo.n_img, p * p * c)
    return pad_rows(x, geo.n_img_pad)


def unpatchify(x: torch.Tensor, geo: Geometry) -> torch.Tensor:
    """[n_img_pad, 64] -> [1, 16, H, W] (diffusers unpatchify, F = pF = 1)."""
    c, p = DIT.in_channels, DIT.patch
    x = x[: geo.n_img].reshape(1, geo.grid_h, geo.grid_w, 1, p, p, c).permute(6, 0, 3, 1, 4, 2, 5)
    return x.reshape(1, c, geo.lat_h, geo.lat_w)


# ----------------------------------------------------------------------------- RoPE
def _axis_freqs(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))


def positions(geo: Geometry):
    """(f, h, w) ids for [image patches (n_img_pad), caption tokens (n_cap_pad)] as in patchify_and_embed."""
    lp = geo.n_cap_pad
    hh = torch.arange(geo.grid_h).repeat_interleave(geo.grid_w)
    ww = torch.arange(geo.grid_w).repeat(geo.grid_h)
    img = torch.stack([torch.full((geo.n_img,), lp + 1), hh, ww], dim=-1)
    img = torch.cat([img, torch.zeros(geo.n_img_pad - geo.n_img, 3, dtype=img.dtype)])
    cap = torch.stack([torch.arange(1, lp + 1), torch.zeros(lp, dtype=torch.long), torch.zeros(lp, dtype=torch.long)], -1)
    return img, cap


def angles(ids: torch.Tensor, cfg: DiTConfig = DIT) -> torch.Tensor:
    """[S, 3] ids -> [S, 64] float32 angles (RopeEmbedder: float64 outer product, then .float())."""
    out = [torch.outer(ids[:, i].to(torch.float64), _axis_freqs(d, cfg.rope_theta)).float()
           for i, d in enumerate(cfg.axes_dims)]
    return torch.cat(out, dim=-1)


def cos_sin(ids: torch.Tensor, dtype=torch.bfloat16):
    """[S, 128] cos/sin, each frequency repeated for its adjacent (real, imag) pair."""
    a = angles(ids)
    return torch.cos(a).repeat_interleave(2, -1).to(dtype), torch.sin(a).repeat_interleave(2, -1).to(dtype)


def te_cos_sin(seq_len: int, dtype=torch.bfloat16):
    """Qwen3 text-encoder tables (positions 0..S-1) in the permuted adjacent-pair layout."""
    ang = torch.outer(torch.arange(seq_len, dtype=torch.float64), _axis_freqs(TE.head_dim, TE.rope_theta)).float()
    return torch.cos(ang).repeat_interleave(2, -1).to(dtype), torch.sin(ang).repeat_interleave(2, -1).to(dtype)


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rot = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- schedule and conditioning
def make_scheduler(snapshot: str):
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(snapshot, subfolder="scheduler")
    sched.sigma_min = 0.0  # ZImagePipeline.__call__ sets this before retrieve_timesteps
    sched.set_timesteps(SCHED.steps)
    return sched


def model_timesteps(sched) -> List[float]:
    """Normalized DiT time inputs (1000 - t) / 1000 of the steps that change the latents.
    The final scheduler step has sigma 0 -> dt 0 and is skipped (latents unchanged)."""
    ts = sched.timesteps.float()
    sig = sched.sigmas.float()
    active = [i for i in range(len(ts)) if float(sig[i + 1] - sig[i]) != 0.0]
    return [float((1000.0 - ts[i]) / 1000.0) for i in active], active


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@dataclass
class StepCond:
    """Per-step folded RMSNorm weights (bf16 [dim]) for every modulated block, plus the final LayerNorm gamma."""

    blocks: List[Dict[str, torch.Tensor]]  # order: noise_refiner.0, noise_refiner.1, layers.0 .. layers.29
    final_gamma: torch.Tensor


def modulated_prefixes(cfg: DiTConfig = DIT) -> List[str]:
    return [f"noise_refiner.{i}" for i in range(cfg.n_refiner_layers)] + [f"layers.{i}" for i in range(cfg.n_layers)]


class TimeConditioning:
    """t -> t_embedder -> adaLN rows, evaluated in the checkpoint dtype (bf16) like the reference, then
    folded: rms(x)*w*(1+s) == rms(x)*(w*(1+s)) and tanh(g)*rms(y)*w == rms(y)*(w*tanh(g))."""

    def __init__(self, ckpt: LazyCheckpoint, cfg: DiTConfig = DIT):
        self.cfg = cfg
        g = lambda k: ckpt.get(k, torch.bfloat16)
        self.t_mlp = [(g(f"t_embedder.mlp.{i}.weight"), g(f"t_embedder.mlp.{i}.bias")) for i in (0, 2)]
        self.blocks = []
        for p in modulated_prefixes(cfg):
            self.blocks.append({
                "mod_w": g(f"{p}.adaLN_modulation.0.weight"), "mod_b": g(f"{p}.adaLN_modulation.0.bias"),
                "attention_norm1": g(f"{p}.attention_norm1.weight"), "attention_norm2": g(f"{p}.attention_norm2.weight"),
                "ffn_norm1": g(f"{p}.ffn_norm1.weight"), "ffn_norm2": g(f"{p}.ffn_norm2.weight"),
            })
        self.final_w = g("all_final_layer.2-1.adaLN_modulation.1.weight")
        self.final_b = g("all_final_layer.2-1.adaLN_modulation.1.bias")

    def adaln_input(self, t_model: float) -> torch.Tensor:
        t = torch.tensor([t_model], dtype=torch.float32) * self.cfg.t_scale
        h = timestep_embedding(t, self.cfg.adaln_dim).to(torch.bfloat16)
        (w0, b0), (w2, b2) = self.t_mlp
        h = torch.nn.functional.linear(h, w0, b0)
        h = torch.nn.functional.silu(h)
        return torch.nn.functional.linear(h, w2, b2)  # [1, 256] bf16

    def step(self, t_model: float) -> StepCond:
        c = self.adaln_input(t_model)
        d = self.cfg.dim
        blocks = []
        for b in self.blocks:
            mod = torch.nn.functional.linear(c, b["mod_w"], b["mod_b"])[0]
            s_msa, g_msa, s_mlp, g_mlp = mod.split(d)
            one_s_msa, one_s_mlp = (1.0 + s_msa).float(), (1.0 + s_mlp).float()
            t_msa, t_mlp = g_msa.tanh().float(), g_mlp.tanh().float()
            blocks.append({
                "attention_norm1": (b["attention_norm1"].float() * one_s_msa).to(torch.bfloat16),
                "attention_norm2": (b["attention_norm2"].float() * t_msa).to(torch.bfloat16),
                "ffn_norm1": (b["ffn_norm1"].float() * one_s_mlp).to(torch.bfloat16),
                "ffn_norm2": (b["ffn_norm2"].float() * t_mlp).to(torch.bfloat16),
            })
        scale = torch.nn.functional.linear(torch.nn.functional.silu(c), self.final_w, self.final_b)[0]
        return StepCond(blocks, (1.0 + scale).to(torch.bfloat16))


# ----------------------------------------------------------------------------- prompt
def tokenize(tokenizer, prompt: str) -> torch.Tensor:
    """Token ids [L] of the chat-formatted prompt, truncated at max_sequence_length like _encode_prompt."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=True)
    ids = tokenizer(text, truncation=True, max_length=TE.max_tokens, return_tensors="pt").input_ids[0]
    return ids


def init_latents(geo: Geometry, seed: int) -> torch.Tensor:
    """prepare_latents: randn (1, 16, lat_h, lat_w) float32 from a CPU generator."""
    g = torch.Generator("cpu").manual_seed(int(seed))
    return torch.randn((1, DIT.in_channels, geo.lat_h, geo.lat_w), generator=g, dtype=torch.float32)
