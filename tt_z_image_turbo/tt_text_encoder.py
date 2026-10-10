# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Z-Image-Turbo text encoder (Qwen3-4B) prefill on one Blackhole, batch 1.

Adapted from the Apache-2.0 Qwen-Image-2.1 single-chip port (Qwen3-VL text stack).
Returns hidden_states[-2] of transformers' Qwen3Model, i.e. the residual stream after 35 of the 36 decoder
layers with no final norm (verified bit-exact on CPU by check_te_layers.py); layer 36 is never loaded.
The tt-metal Z-Image demo runs all 36 layers, which is a different tensor (PCC 0.991, max |diff| 2240).
q/k projection rows are permuted per head so the adjacent-pair RoPE kernel reproduces rotate_half.
Token embeddings are gathered on host (the 0.78 GB table stays off-device). Trailing padding to a tile
multiple cannot affect the valid rows under causal attention.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import ttnn

from . import host
from .config import TE, TILE, TextEncoderConfig

MEM = ttnn.DRAM_MEMORY_CONFIG
PFX = "model."


@dataclass
class TEPrecision:
    weight_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2


class TELayer:
    def __init__(self, dev, ckpt: host.LazyCheckpoint, idx: int, prec: TEPrecision, perm, cfg: TextEncoderConfig):
        g = lambda k: ckpt.get(f"{PFX}layers.{idx}.{k}", torch.bfloat16)
        wd = prec.weight_dtype
        dt = lambda t, dtype=wd: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=MEM)
        wq = host.permute_heads_rows(g("self_attn.q_proj.weight"), cfg.heads, cfg.head_dim, perm)
        wk = host.permute_heads_rows(g("self_attn.k_proj.weight"), cfg.kv_heads, cfg.head_dim, perm)
        self.wqkv = dt(host.fuse_qkv(wq, wk, g("self_attn.v_proj.weight")))
        row = lambda t: dt(t.reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.q_norm = row(g("self_attn.q_norm.weight")[perm])
        self.k_norm = row(g("self_attn.k_norm.weight")[perm])
        self.wo = dt(host.linear_to_mm(g("self_attn.o_proj.weight")))
        self.w_gateup = dt(host.swiglu_interleave(g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")))
        self.w_down = dt(host.linear_to_mm(g("mlp.down_proj.weight")))
        self.ln1 = row(g("input_layernorm.weight"))
        self.ln2 = row(g("post_attention_layernorm.weight"))


class ZImageTextEncoder:
    def __init__(self, dev, ckpt: host.LazyCheckpoint, prec: Optional[TEPrecision] = None,
                 cfg: TextEncoderConfig = TE):
        self.dev, self.cfg, self.ckpt = dev, cfg, ckpt
        self.prec = prec or TEPrecision()
        self.perm = host.interleave_pairs_permutation(cfg.head_dim)
        self.layers: List[TELayer] = [TELayer(dev, ckpt, i, self.prec, self.perm, cfg) for i in range(cfg.num_layers)]
        self.grid = dev.compute_with_storage_grid_size()
        arch = dev.arch()
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=self.prec.mm_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
        self.ck_norm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True,
            packer_l1_acc=False)
        self.trans_mat = ttnn.from_torch(host.rot_transformation_mat(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=MEM)

    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.ck_mm, memory_config=MEM, dtype=ttnn.bfloat16)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.cfg.rms_eps, weight=w, compute_kernel_config=self.ck_norm,
                             memory_config=MEM)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[L] token ids -> [L, 2560] bf16 host caption features (hidden_states[-2])."""
        c = self.cfg
        ids = input_ids.reshape(-1)
        L = ids.shape[0]
        Sp = host.round_up(L, TILE)
        emb = self.ckpt.get_rows(PFX + "embed_tokens.weight", ids).to(torch.bfloat16)
        h = ttnn.from_torch(host.pad_rows(emb, Sp).reshape(1, 1, Sp, -1), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=self.dev, memory_config=MEM)
        cos_h, sin_h = host.te_cos_sin(Sp)
        tab = lambda t: ttnn.from_torch(t.reshape(1, 1, Sp, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                        device=self.dev, memory_config=MEM)
        cos, sin = tab(cos_h), tab(sin_h)
        sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=self.grid, q_chunk_size=TILE,
                                          k_chunk_size=TILE, exp_approx_mode=False)
        for layer in self.layers:
            x = self._rms(h, layer.ln1)
            qkv = self._linear(x, layer.wqkv)
            ttnn.deallocate(x)
            if len(qkv.shape) != 4:
                qkv = ttnn.reshape(qkv, [1, 1, Sp, -1])
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=c.heads, num_kv_heads=c.kv_heads,
                                                             transpose_k_heads=False, memory_config=MEM)
            ttnn.deallocate(qkv)
            rot = []
            for t, w in ((q, layer.q_norm), (k, layer.k_norm)):
                n = self._rms(t, w)
                ttnn.deallocate(t)
                rot.append(ttnn.experimental.rotary_embedding_llama(n, cos, sin, self.trans_mat, is_decode_mode=False,
                                                                    compute_kernel_config=self.ck_norm))
                ttnn.deallocate(n)
            attn = ttnn.transformer.scaled_dot_product_attention(
                rot[0], rot[1], v, is_causal=True, scale=c.head_dim ** -0.5, program_config=sdpa_cfg,
                compute_kernel_config=self.ck_mm, memory_config=MEM)
            for t in (*rot, v):
                ttnn.deallocate(t)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=MEM)
            ttnn.deallocate(attn)
            o = self._linear(a, layer.wo)
            ttnn.deallocate(a)
            h2 = ttnn.add(h, o, memory_config=MEM)
            ttnn.deallocate(o)
            ttnn.deallocate(h)
            x2 = self._rms(h2, layer.ln2)
            m = ttnn.experimental.minimal_matmul(x2, layer.w_gateup, compute_kernel_config=self.ck_mm,
                                                 dtype=ttnn.bfloat16, memory_config=MEM, fuse_swiglu=True)
            ttnn.deallocate(x2)
            d = self._linear(m, layer.w_down)
            ttnn.deallocate(m)
            h = ttnn.add(h2, d, memory_config=MEM)
            ttnn.deallocate(d)
            ttnn.deallocate(h2)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        out = ttnn.to_torch(h)[0, 0, :L].to(torch.bfloat16)
        ttnn.deallocate(h)
        return out
