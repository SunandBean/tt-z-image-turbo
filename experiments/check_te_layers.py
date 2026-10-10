"""Which Qwen3 layer output does Z-Image's hidden_states[-2] correspond to? CPU only.

Runs the Z-Image-Turbo text encoder with output_hidden_states=True and compares
hidden_states[-2] with the raw residual stream after 35 and after 36 decoder layers.
The upstream TT demo feeds the 36-layer output; diffusers feeds hidden_states[-2]."""
import json
import os
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

SNAPSHOT = Path(os.environ["Z_IMAGE_SNAPSHOT"])
PROMPT = "a misty mountain lake at dawn, watercolor painting"


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def main():
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    tok = AutoTokenizer.from_pretrained(SNAPSHOT / "tokenizer")
    text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    ids = tok(text, return_tensors="pt").input_ids
    model = AutoModel.from_pretrained(SNAPSHOT / "text_encoder", torch_dtype=torch.bfloat16).eval()

    captured = []
    hooks = [layer.register_forward_hook(lambda m, i, o: captured.append((o[0] if isinstance(o, tuple) else o).detach()))
             for layer in model.layers]
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    for h in hooks:
        h.remove()

    hs = out.hidden_states
    ref = hs[-2][0].float()
    after35, after36 = captured[34][0].float(), captured[35][0].float()
    report = {
        "transformers": __import__("transformers").__version__,
        "tokens": ids.shape[1],
        "num_hidden_states": len(hs),
        "max_abs_vs_after35": float((ref - after35).abs().max()),
        "max_abs_vs_after36": float((ref - after36).abs().max()),
        "pcc_vs_after35": pcc(ref, after35),
        "pcc_vs_after36": pcc(ref, after36),
        "pcc_after35_vs_after36": pcc(after35, after36),
        "last_is_normed_after36": float((hs[-1][0].float() - model.norm(captured[35])[0].float()).abs().max()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
