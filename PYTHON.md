# tt_z_image_turbo — Python reference

Z-Image-Turbo text-to-image on one Tenstorrent Blackhole p100a. The Qwen3-4B text encoder,
the DiT and the VAE all run on the card; the scheduler and the image post-processing are host.

## Install

```bash
pip install -e .                # inside a tt-metal / ttnn environment; ttnn is not on PyPI
pip install -e ".[server]"      # also fastapi / uvicorn / pydantic
```

## Weights

`Tongyi-MAI/Z-Image-Turbo`, pinned to `f332072aa78be7aecdf3ee76d5c247082da564a6`. It is downloaded
into the HF cache on first use. `Z_IMAGE_SNAPSHOT` points at a snapshot directory you manage instead.

## API

### `open_device(trace_region_size=0, l1_small_size=98304)`

Opens a 1×1 mesh with `DispatchCoreType.WORKER`. Pair with `close_device(dev)`.

### `ZImageTurboTT(dev, snapshot=None, dit_prec=None, te_prec=None, tt_vae=True)`

Loads the text encoder, the DiT and (unless `tt_vae=False`) the VAE onto the card. With
`tt_vae=False` the diffusers VAE runs on the host in bf16 instead. `.load_s` reports the load time.

### `ZImageTurboTT.generate(prompt, width, height, seed) -> (PIL.Image, Timing)`

| Argument | Meaning |
|---|---|
| `prompt` | Text prompt. |
| `width`, `height` | 512–1920, multiples of 8, at most 1920×1088 pixels. Non-multiples of 16 are generated at the next multiple of 16 and center-cropped. |
| `seed` | Same prompt, size and seed reproduce the image pixel for pixel. |

`Timing.values` holds `text_encoder_s`, `caption_s`, `dit_s`, `vae_decode_s`.

The steps are fixed at 9 scheduler steps (8 DiT forwards; the last sigma is 0).

### Lower-level entry points

| Method | |
|---|---|
| `encode(prompt)` | caption features alone, if you want to reuse them across seeds |
| `denoise(cap_feats, width, height, seed, timing)` | the DiT loop, returning latents |
| `decode(latents)` | VAE decode, returning `(PIL.Image, tensor)` |

### `dram_stats(dev) -> dict`

`total_bytes`, `allocated_bytes`, `free_bytes`, `largest_free_bytes_per_bank`.

## Modules

| Module | Role |
|---|---|
| `pipeline.py` | `ZImageTurboTT`, device open/close, timing |
| `host.py` | host maths: tokenization, scheduler, latent init, patchify/unpatchify, and the shared checkpoint and tensor helpers |
| `tt_dit.py` | the DiT on the card |
| `tt_text_encoder.py` | Qwen3-4B (35 layers) on the card |
| `tt_vae.py` | the VAE on the card, over `vendor_tt_dit` |
| `vendor_tt_dit/` | `tt_dit` VAE from tt-metal `01d6e7b`, with non-square splitting added |
| `ref_algo.py` | torch reference implementations used by the verification scripts |
| `config.py` | static shapes, the pinned revision, snapshot resolution |
| `server.py` | the HTTP app |
