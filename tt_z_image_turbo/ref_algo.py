# SPDX-License-Identifier: Apache-2.0
"""Torch implementation of exactly the algorithm the TT port runs (folded norms, adjacent-pair RoPE,
caption path computed once per prompt, pad-token substitution). Used on CPU to prove the host-side
transformations reproduce diffusers before any device run; the TT DiT mirrors it op for op."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import host
from .config import DIT, DiTConfig


def rms(x, w, eps):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float()).to(torch.bfloat16)


class RefDiT:
    def __init__(self, ckpt: host.LazyCheckpoint, cfg: DiTConfig = DIT):
        self.cfg = cfg
        self.g = lambda k: ckpt.get(k, torch.bfloat16)
        self.time = host.TimeConditioning(ckpt, cfg)

    def _attn(self, x, p, cos, sin):
        c, g = self.cfg, self.g
        S = x.shape[0]
        q = F.linear(x, g(f"{p}.attention.to_q.weight")).view(S, c.heads, c.head_dim)
        k = F.linear(x, g(f"{p}.attention.to_k.weight")).view(S, c.heads, c.head_dim)
        v = F.linear(x, g(f"{p}.attention.to_v.weight")).view(S, c.heads, c.head_dim)
        q = rms(q, g(f"{p}.attention.norm_q.weight"), c.norm_eps)
        k = rms(k, g(f"{p}.attention.norm_k.weight"), c.norm_eps)
        q = host.apply_rope_adjacent(q.float(), cos[:, None].float(), sin[:, None].float()).to(torch.bfloat16)
        k = host.apply_rope_adjacent(k.float(), cos[:, None].float(), sin[:, None].float()).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return F.linear(o.transpose(0, 1).reshape(S, -1), g(f"{p}.attention.to_out.0.weight"))

    def _ffn(self, x, p):
        g = self.g
        return F.linear(F.silu(F.linear(x, g(f"{p}.feed_forward.w1.weight"))) * F.linear(x, g(f"{p}.feed_forward.w3.weight")),
                        g(f"{p}.feed_forward.w2.weight"))

    def _block(self, x, p, cos, sin, norms):
        eps = self.cfg.norm_eps
        x = x + rms(self._attn(rms(x, norms["attention_norm1"], eps), p, cos, sin), norms["attention_norm2"], eps)
        return x + rms(self._ffn(rms(x, norms["ffn_norm1"], eps), p), norms["ffn_norm2"], eps)

    def static_norms(self, p):
        return {k: self.g(f"{p}.{k}.weight") for k in ("attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2")}

    def caption(self, cap_feats: torch.Tensor, geo: host.Geometry) -> torch.Tensor:
        """[L, 2560] -> refined caption tokens [n_cap_pad, dim] (once per prompt)."""
        c, g = self.cfg, self.g
        x = rms(host.pad_rows(cap_feats.to(torch.bfloat16), geo.n_cap_pad), g("cap_embedder.0.weight"), c.norm_eps)
        x = F.linear(x, g("cap_embedder.1.weight"), g("cap_embedder.1.bias"))
        x[geo.cap_len:] = g("cap_pad_token")[0]
        _, cap_ids = host.positions(geo)
        cos, sin = host.cos_sin(cap_ids)
        for i in range(c.n_refiner_layers):
            p = f"context_refiner.{i}"
            x = self._block(x, p, cos, sin, self.static_norms(p))
        return x

    def step(self, patches: torch.Tensor, cap: torch.Tensor, cond: host.StepCond, geo: host.Geometry,
             taps=None) -> torch.Tensor:
        """[n_img_pad, 64] patches -> [n_img_pad, 64] model output for one timestep."""
        c, g = self.cfg, self.g
        x = F.linear(patches.to(torch.bfloat16), g("all_x_embedder.2-1.weight"), g("all_x_embedder.2-1.bias"))
        x[geo.n_img:] = g("x_pad_token")[0]
        img_ids, cap_ids = host.positions(geo)
        cos_i, sin_i = host.cos_sin(img_ids)
        cos_j, sin_j = host.cos_sin(torch.cat([img_ids, cap_ids]))
        prefixes = host.modulated_prefixes(c)
        tap = (lambda t: taps.append(t.float().clone())) if taps is not None else (lambda t: None)
        tap(x)
        for i in range(c.n_refiner_layers):
            x = self._block(x, prefixes[i], cos_i, sin_i, cond.blocks[i])
            tap(x)
        u = torch.cat([x, cap])
        for i in range(c.n_refiner_layers, len(prefixes)):
            u = self._block(u, prefixes[i], cos_j, sin_j, cond.blocks[i])
            tap(u)
        u = u[: geo.n_img_pad]
        u = F.layer_norm(u.float(), (c.dim,), eps=c.final_eps).to(torch.bfloat16) * cond.final_gamma
        return F.linear(u, g("all_final_layer.2-1.linear.weight"), g("all_final_layer.2-1.linear.bias"))
