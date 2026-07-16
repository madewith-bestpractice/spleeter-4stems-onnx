#!/usr/bin/env python3
"""Export the four ported U-Nets to ONNX, fp32 + fp16.

sherpa-onnx's export_onnx.py hardcodes 2stems and two stem names; this is the
same thing (opset 13, dynamic `num_splits`, keep_io_types on the fp16 pass) for
four, with the ELU-configured U-Net.
"""

import onnx
import onnxmltools
import torch
from onnxconverter_common.float16 import convert_float_to_float16

from unet_any import UNet

STEMS = ("vocals", "drums", "bass", "other")
OUT = "./4stems"


def add_meta(filename: str, stem: str) -> None:
    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for k, v in {
        "model_type": "spleeter",
        "sample_rate": 44100,
        "version": 1,
        "model_url": "https://github.com/deezer/spleeter",
        "stems": 4,
        "comment": stem,
        "model_name": "4stems.tar.gz",
        # The bit that is not obvious and that a future reader will need:
        "conv_activation": "ELU",
        "deconv_activation": "ELU",
    }.items():
        m = model.metadata_props.add()
        m.key, m.value = k, str(v)
    onnx.save(model, filename)


@torch.no_grad()
def main() -> None:
    for stem in STEMS:
        net = UNet("ELU", "ELU")
        net.load_state_dict(torch.load(f"{OUT}/{stem}.pt", map_location="cpu"))
        net.eval()

        x = torch.rand(2, 1, 512, 1024, dtype=torch.float32)
        f32 = f"{OUT}/{stem}.onnx"
        torch.onnx.export(
            net, x, f32,
            input_names=["x"], output_names=["y"],
            dynamic_axes={"x": {1: "num_splits"}},
            opset_version=13,
        )
        add_meta(f32, stem)

        f16 = f"{OUT}/{stem}.fp16.onnx"
        onnxmltools.utils.save_model(
            convert_float_to_float16(
                onnxmltools.utils.load_model(f32), keep_io_types=True
            ),
            f16,
        )
        print(f"  {stem:<8} -> {f32}  +  {f16}")


if __name__ == "__main__":
    main()
