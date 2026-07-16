#!/usr/bin/env python3
"""Which tensor actually enters the U-Net?

convert_to_torch.py hardcodes `strided_slice_3:0` as the spectrogram feeding the
net. That name was read off the *2stems* graph, and TF numbers ops across the
whole graph -- so there is no reason it should mean the same thing in a graph
built with four U-Nets instead of two. The torch port matches sherpa's mapping
exactly and still fails by 945, which points here rather than at the weights.

So: don't guess the name. Walk back from the stem's first conv2d op to whatever
feeds it, and print every StridedSlice with its shape for comparison.
"""

import sys

import tensorflow as tf

from convert_any import family_of, index_of, load_graph, scopes


def main(path: str, stem: str) -> None:
    g = load_graph(path)

    first = scopes(g, "conv2d", "/kernel")[0]
    conv_op = g.get_operation_by_name(f"{first}/Conv2D")
    print(f"first conv of {stem}: {conv_op.name}")

    # Walk back until we leave the padding/reshape plumbing.
    t = conv_op.inputs[0]
    for _ in range(6):
        op = t.op
        print(f"  <- {op.name:<40} {op.type:<14} {t.shape}")
        if op.type in ("StridedSlice", "Abs", "Reshape"):
            break
        if not op.inputs:
            break
        t = op.inputs[0]

    print("\nAll top-level StridedSlice outputs:")
    for op in g.get_operations():
        if op.type == "StridedSlice" and "/" not in op.name:
            print(f"  {op.name:<20} {op.outputs[0].shape}")

    print("\nThe stem output:")
    mul = g.get_operation_by_name(f"{stem}_spectrogram/mul")
    print(f"  {mul.name:<20} {mul.outputs[0].shape}")
    for i in mul.inputs:
        print(f"     input <- {i.op.name:<32} {i.op.type:<12} {i.shape}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
