# tt-z-image-turbo

**What this port adds** — the single-chip port itself, non-square splitting in the `tt_dit` VAE (which assumed square latents), and the standalone HTTP server and Hub weight resolution in `code/`.
**What it builds on** — the [`changh95/qwen-image-2.1-p150`](https://huggingface.co/changh95/qwen-image-2.1-p150) single-chip port, the [tt-metal](https://github.com/tenstorrent/tt-metal) Z-Image demo and its `tt_dit` VAE, all Apache-2.0.

Tongyi-MAI **Z-Image-Turbo** ported to a single **Tenstorrent Blackhole p100a**. The Qwen3-4B text
encoder, the DiT and the VAE all run on the card in TTNN.

Model card and demos: **[sunandbean/z-image-turbo-p100a](https://huggingface.co/sunandbean/z-image-turbo-p100a)**

| 1024×1024, 7.76 s | 576×1024 |
|:---:|:---:|
| ![](media/teapot1024.png) | ![](media/fox576x1024.png) |

A 1024×1024 image takes **7.76 s** warm (DiT 6.24 s, VAE 1.20 s); 512×512 takes 1.44 s. The same
prompt, size and seed reproduce the image pixel for pixel.

Against an RTX 5070 Ti running the diffusers `ZImagePipeline` at bf16, same prompt, size, seed and
9-step schedule: **0.776 s per DiT forward on the card against 1.578 s on the GPU**. End to end the
gap is wider (7.68 s against 22.8 s) but that is partly a memory story — the model does not fit in
15.5 GB, so the GPU run offloads stage by stage. Numbers and method on the model card.

## Install

```bash
pip install -e .    # inside a tt-metal / ttnn environment; ttnn is not on PyPI
```

```python
from tt_z_image_turbo import ZImageTurboTT, open_device, close_device

dev = open_device()
try:
    model = ZImageTurboTT(dev)      # downloads the weights into the HF cache on first use
    image, timing = model.generate("A croissant beside a cup of coffee, food photograph",
                                   1024, 1024, seed=42)
    image.save("output.png")
finally:
    close_device(dev)
```

Or over HTTP: `uvicorn tt_z_image_turbo.server:app --host 0.0.0.0 --port 20000`.

## Layout

| Path | |
|---|---|
| `tt_z_image_turbo/` | the port |
| `tt_z_image_turbo/vendor_tt_dit/` | `tt_dit` VAE from tt-metal `01d6e7b`, with non-square splitting added |
| `examples/quickstart.py` | runnable example |
| `tt-model.yaml` | the container manifest the published image is built from — `tt-model package --container tt-model.yaml` |
| `PYTHON.md` | API reference |
| `experiments/` | the verification scripts behind the published numbers — most import this port under the name it had in the private tree it was written in, so read [`experiments/README.md`](experiments/README.md) before running them |

## Note on the tt-metal demo

The official tt-metal Z-Image demo reads the text encoder's **36th**-layer output and pads captions to
a fixed 128 tokens, neither of which matches upstream Z-Image. This port uses the 35th layer and the
real caption length, so it follows the original model rather than the demo.

## Licence

Apache-2.0. The weights ([Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo),
Apache-2.0) are not redistributed here.
