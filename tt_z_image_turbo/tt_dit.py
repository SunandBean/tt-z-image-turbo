# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Z-Image-Turbo DiT on ONE Blackhole (P100a), TTNN, batch 1. Mirrors ref_algo.RefDiT op for op.

Adapted from Tenstorrent's Apache-2.0 code: the Qwen-Image-2.1 single-chip port (changh95/qwen-image-2.1-p150:
fused QKV + per-head RMSNorm + adjacent-pair rotary_embedding_llama, fused SwiGLU minimal_matmul) and the
tt-metal Z-Image-Turbo demo (models/demos/z_image_turbo, 4-chip TP=4) for the Z-Image block structure.

Differences from the tt-metal demo, all needed for one chip and for parity with diffusers:
  * TP = 1: 30 unpadded heads, no CCL, and no duplicate (pre-transpose) copies of every weight
  * any 16 px grid size: image tokens and caption tokens are padded to 32 like diffusers, RoPE tables built on
    host once per geometry (the demo rebuilt them inside every attention call, fixed 512 x 512 / 128 tokens)
  * caption tokens = valid prompt tokens only (the demo fed 128 max-length tokens incl. padding)
  * adaLN is precomputed for the fixed 9-step schedule and folded into the RMSNorm / LayerNorm weights; the
    context refiner (timestep-free) runs once per prompt instead of once per step
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import ttnn

from . import host
from .config import DIT, TILE, DiTConfig

try:
    from models.tt_dit.utils.matmul import get_matmul_config
except ImportError:  # pragma: no cover - only on hosts without tt-metal models/
    get_matmul_config = None

MEM = ttnn.DRAM_MEMORY_CONFIG
NORM_KEYS = ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2")


@dataclass
class DiTPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2
    # diag_precision.py: HiFi4 + fp32-accumulated SDPA lifts 848x624 8-step latents PCC 0.918 -> 0.948 (512^2:
    # 0.998 -> 0.9986) at +3 % step time; HiFi4 matmuls cost +40 % and did not improve the image
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4
    sdpa_fp32_acc: bool = True


def _dev_tensor(dev, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev, memory_config=MEM)


def _row(dev, v: torch.Tensor):
    """[D] -> [1, 1, 1, D] bf16 tile tensor (norm gamma / bias row)."""
    return _dev_tensor(dev, v.to(torch.bfloat16).reshape(1, 1, 1, -1))


class Block:
    def __init__(self, dev, ckpt: host.LazyCheckpoint, prefix: str, prec: DiTPrecision, cfg: DiTConfig):
        g = lambda k: ckpt.get(f"{prefix}.{k}", torch.bfloat16)
        wd = prec.weight_dtype
        self.wqkv = _dev_tensor(dev, host.fuse_qkv(g("attention.to_q.weight"), g("attention.to_k.weight"),
                                                   g("attention.to_v.weight")), wd)
        self.norm_q = _row(dev, g("attention.norm_q.weight"))
        self.norm_k = _row(dev, g("attention.norm_k.weight"))
        self.wo = _dev_tensor(dev, host.linear_to_mm(g("attention.to_out.0.weight")), wd)
        self.w_gateup = _dev_tensor(dev, host.swiglu_interleave(g("feed_forward.w1.weight"), g("feed_forward.w3.weight")), wd)
        self.w_down = _dev_tensor(dev, host.linear_to_mm(g("feed_forward.w2.weight")), wd)
        self.static_norms = None
        if "context_refiner" in prefix:  # timestep-free block: its own RMSNorm weights, uploaded once
            self.static_norms = {k: _row(dev, g(f"{k}.weight")) for k in NORM_KEYS}


@dataclass
class Tables:
    cos: ttnn.Tensor
    sin: ttnn.Tensor


@dataclass
class Prepared:
    """Device state for one (width, height, caption length) geometry."""

    geo: host.Geometry
    rope_img: Tables
    rope_joint: Tables
    rope_cap: Tables
    x_pad_delta: ttnn.Tensor  # [1,1,n_img_pad,dim]: x_pad_token - bias on pad rows, 0 elsewhere


class ZImageDiT:
    def __init__(self, dev, ckpt: host.LazyCheckpoint, prec: Optional[DiTPrecision] = None, cfg: DiTConfig = DIT):
        self.dev, self.cfg = dev, cfg
        self.prec = prec or DiTPrecision()
        self.grid = dev.compute_with_storage_grid_size()
        g = lambda k: ckpt.get(k, torch.bfloat16)
        self.refiners = [Block(dev, ckpt, f"noise_refiner.{i}", self.prec, cfg) for i in range(cfg.n_refiner_layers)]
        self.context = [Block(dev, ckpt, f"context_refiner.{i}", self.prec, cfg) for i in range(cfg.n_refiner_layers)]
        self.layers = [Block(dev, ckpt, f"layers.{i}", self.prec, cfg) for i in range(cfg.n_layers)]
        self.x_w = _dev_tensor(dev, host.linear_to_mm(g("all_x_embedder.2-1.weight")))
        self.x_b = _row(dev, g("all_x_embedder.2-1.bias"))
        self._x_bias_host = g("all_x_embedder.2-1.bias")
        self._x_pad_host = g("x_pad_token")[0]
        self.cap_norm_w = g("cap_embedder.0.weight")  # caption embed runs on host once per prompt (exact pad rows)
        self.cap_w = g("cap_embedder.1.weight")
        self.cap_b = g("cap_embedder.1.bias")
        self.cap_pad = g("cap_pad_token")[0]
        self.final_w = _dev_tensor(dev, host.linear_to_mm(g("all_final_layer.2-1.linear.weight")))
        self.final_b = _row(dev, g("all_final_layer.2-1.linear.bias"))
        self.trans_mat = _dev_tensor(dev, host.rot_transformation_mat())

        arch = dev.arch()
        ck = lambda fid, fp32: ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False)
        self.ck_mm = ck(self.prec.mm_fidelity, True)
        self.ck_norm = ck(ttnn.MathFidelity.HiFi4, True)
        self.ck_sdpa = ck(self.prec.sdpa_fidelity, self.prec.sdpa_fp32_acc)
        self.ck_rope = ck(ttnn.MathFidelity.HiFi4, True)
        self._mm_cfg_cache: Dict[tuple, object] = {}
        self.sdpa_chunks = (256, 128, 64, 32)  # largest dividing S is used; diagnostics may narrow it

        # adaLN of the fixed schedule, folded into norm weights, resident for every step
        self.time = host.TimeConditioning(ckpt, cfg)
        self.step_conds: Dict[float, List[Dict[str, ttnn.Tensor]]] = {}
        self.step_final: Dict[float, ttnn.Tensor] = {}

    # ------------------------------------------------------------------ conditioning
    def load_schedule(self, t_models: List[float]):
        for t in t_models:
            if t in self.step_conds:
                continue
            c = self.time.step(t)
            self.step_conds[t] = [{k: _row(self.dev, b[k]) for k in NORM_KEYS} for b in c.blocks]
            self.step_final[t] = _row(self.dev, c.final_gamma)

    # ------------------------------------------------------------------ geometry
    def _tables(self, ids: torch.Tensor) -> Tables:
        cos, sin = host.cos_sin(ids)
        f = lambda t: _dev_tensor(self.dev, t.reshape(1, 1, t.shape[0], -1))
        return Tables(f(cos), f(sin))

    def prepare(self, geo: host.Geometry) -> Prepared:
        img_ids, cap_ids = host.positions(geo)
        delta = torch.zeros(geo.n_img_pad, self.cfg.dim, dtype=torch.float32)
        delta[geo.n_img:] = self._x_pad_host.float() - self._x_bias_host.float()
        return Prepared(geo, self._tables(img_ids), self._tables(torch.cat([img_ids, cap_ids])), self._tables(cap_ids),
                        _dev_tensor(self.dev, delta.to(torch.bfloat16).reshape(1, 1, geo.n_img_pad, -1)))

    def release(self, prep: Prepared):
        for t in (prep.rope_img, prep.rope_joint, prep.rope_cap):
            ttnn.deallocate(t.cos)
            ttnn.deallocate(t.sin)
        ttnn.deallocate(prep.x_pad_delta)

    # ------------------------------------------------------------------ ops
    def _mm_cfg(self, M, K, N):
        key = (M, K, N)
        if key not in self._mm_cfg_cache:
            self._mm_cfg_cache[key] = get_matmul_config(M, K, N, self.grid)
        return self._mm_cfg_cache[key]

    def _mm(self, x, w, M, K, N, fuse_swiglu=False):
        return ttnn.experimental.minimal_matmul(x, w, config=self._mm_cfg(M, K, N), compute_kernel_config=self.ck_mm,
                                                dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=fuse_swiglu)

    def _rms(self, x, w, eps=None):
        return ttnn.rms_norm(x, epsilon=eps or self.cfg.norm_eps, weight=w, compute_kernel_config=self.ck_norm,
                             memory_config=MEM)

    def _sdpa_cfg(self, S):
        chunk = next(c for c in self.sdpa_chunks if S % c == 0)
        return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=chunk,
                                      k_chunk_size=chunk, exp_approx_mode=False)

    def _attention(self, h, blk: Block, S: int, rt: Tables):
        c = self.cfg
        qkv = self._mm(h, blk.wqkv, S, c.dim, 3 * c.dim)
        if len(qkv.shape) != 4:
            qkv = ttnn.reshape(qkv, [1, 1, S, 3 * c.dim])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.heads,
                                                         transpose_k_heads=False, memory_config=MEM)
        ttnn.deallocate(qkv)
        outs = []
        for t, w in ((q, blk.norm_q), (k, blk.norm_k)):
            n = self._rms(t, w)
            ttnn.deallocate(t)
            outs.append(ttnn.experimental.rotary_embedding_llama(n, rt.cos, rt.sin, self.trans_mat, is_decode_mode=False,
                                                                 compute_kernel_config=self.ck_rope))
            ttnn.deallocate(n)
        attn = ttnn.transformer.scaled_dot_product_attention(
            outs[0], outs[1], v, is_causal=False, scale=c.head_dim ** -0.5, program_config=self._sdpa_cfg(S),
            compute_kernel_config=self.ck_sdpa, memory_config=MEM)
        for t in (*outs, v):
            ttnn.deallocate(t)
        a = ttnn.transformer.concatenate_heads(attn, memory_config=MEM)
        ttnn.deallocate(attn)
        o = self._mm(a, blk.wo, S, c.dim, c.dim)
        ttnn.deallocate(a)
        return o

    def _block(self, x, blk: Block, S: int, rt: Tables, norms: Dict[str, ttnn.Tensor]):
        """x + rms(attn(rms(x)*n1))*n2, then x + rms(ffn(rms(x)*n3))*n4 with folded gamma rows."""
        c = self.cfg
        h = self._rms(x, norms["attention_norm1"])
        o = self._attention(h, blk, S, rt)
        ttnn.deallocate(h)
        on = self._rms(o, norms["attention_norm2"])
        ttnn.deallocate(o)
        x2 = ttnn.add(x, on, memory_config=MEM)
        ttnn.deallocate(on)
        ttnn.deallocate(x)
        h = self._rms(x2, norms["ffn_norm1"])
        m = self._mm(h, blk.w_gateup, S, c.dim, 2 * c.mlp_hidden, fuse_swiglu=True)
        ttnn.deallocate(h)
        d = self._mm(m, blk.w_down, S, c.mlp_hidden, c.dim)
        ttnn.deallocate(m)
        dn = self._rms(d, norms["ffn_norm2"])
        ttnn.deallocate(d)
        out = ttnn.add(x2, dn, memory_config=MEM)
        ttnn.deallocate(dn)
        ttnn.deallocate(x2)
        return out

    # ------------------------------------------------------------------ public
    def caption(self, cap_feats: torch.Tensor, prep: Prepared) -> ttnn.Tensor:
        """[L, 2560] text features -> refined caption tokens [1,1,n_cap_pad,dim] on device (once per prompt)."""
        geo, c = prep.geo, self.cfg
        x = host.pad_rows(cap_feats.to(torch.bfloat16), geo.n_cap_pad)
        xf = x.float()
        x = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + c.norm_eps) * self.cap_norm_w.float()).to(torch.bfloat16)
        x = torch.nn.functional.linear(x, self.cap_w, self.cap_b)
        x[geo.cap_len:] = self.cap_pad
        h = _dev_tensor(self.dev, x.reshape(1, 1, geo.n_cap_pad, -1))
        for blk in self.context:
            h = self._block(h, blk, geo.n_cap_pad, prep.rope_cap, blk.static_norms)
        return h

    def step(self, patches: torch.Tensor, cap: ttnn.Tensor, t_model: float, prep: Prepared,
             taps: Optional[list] = None) -> torch.Tensor:
        """[n_img_pad, 64] host patches -> [n_img_pad, 64] float32 host model output (velocity, pre-negation)."""
        geo, c = prep.geo, self.cfg
        norms, final_gamma = self.step_conds[t_model], self.step_final[t_model]
        p = _dev_tensor(self.dev, patches.to(torch.bfloat16).reshape(1, 1, geo.n_img_pad, -1))
        x0 = ttnn.linear(p, self.x_w, bias=self.x_b, compute_kernel_config=self.ck_mm, memory_config=MEM,
                         dtype=ttnn.bfloat16)
        ttnn.deallocate(p)
        x = ttnn.add(x0, prep.x_pad_delta, memory_config=MEM)
        ttnn.deallocate(x0)
        tap = (lambda t: taps.append(ttnn.to_torch(t)[0, 0].float())) if taps is not None else (lambda t: None)
        tap(x)
        for i, blk in enumerate(self.refiners):
            x = self._block(x, blk, geo.n_img_pad, prep.rope_img, norms[i])
            tap(x)
        u = ttnn.concat([x, cap], dim=2, memory_config=MEM)
        ttnn.deallocate(x)
        base = len(self.refiners)
        for i, blk in enumerate(self.layers):
            u = self._block(u, blk, geo.n_joint, prep.rope_joint, norms[base + i])
            tap(u)
        xi = ttnn.slice(u, [0, 0, 0, 0], [1, 1, geo.n_img_pad, c.dim], memory_config=MEM)
        ttnn.deallocate(u)
        xn = ttnn.layer_norm(xi, epsilon=c.final_eps, weight=final_gamma, compute_kernel_config=self.ck_norm,
                             memory_config=MEM)
        ttnn.deallocate(xi)
        out = ttnn.linear(xn, self.final_w, bias=self.final_b, compute_kernel_config=self.ck_mm, memory_config=MEM,
                          dtype=ttnn.bfloat16)
        ttnn.deallocate(xn)
        host_out = ttnn.to_torch(out)[0, 0, : geo.n_img_pad].float()
        ttnn.deallocate(out)
        return host_out
