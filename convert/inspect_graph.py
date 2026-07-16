#!/usr/bin/env python3
"""What is actually in a frozen 4stems graph?

sherpa-onnx's convert_to_torch.py hardcodes TF op-name offsets for exactly two
stems (vocals -> conv2d/*, accompaniment -> conv2d_7/*, bn_offset=12). To reach
drums/bass/other, the mapping has to be derived rather than guessed -- and there
is a trap in it: within one stem's block the BatchNorm indices run
0,1,2,3,4,*6*,7,8,9,10,11. Index 5 is skipped, because the U-Net has 11
BatchNorms but TF allocated 12 per stem (conv5 has none).

So "collect and sort" only reproduces the right mapping if freezing prunes the
unused BatchNorm. This prints what survives, per stem, so the answer is
observed instead of assumed.
"""

import re
import sys

import tensorflow as tf


def index_of(name: str) -> int:
    """conv2d/kernel -> 0, conv2d_7/kernel -> 7."""
    m = re.match(r"^[a-z_2]+?(?:_(\d+))?/", name)
    return int(m.group(1)) if m and m.group(1) else 0


def load(path):
    with tf.compat.v1.gfile.GFile(path, "rb") as f:
        gd = tf.compat.v1.GraphDef()
        gd.ParseFromString(f.read())
    with tf.Graph().as_default() as g:
        tf.import_graph_def(gd, name="")
    return g


def main(path):
    g = load(path)
    consts = [op.name for op in g.get_operations() if op.type == "Const"]

    for family, suffix in (
        ("conv2d_transpose", "/kernel"),   # before conv2d: it is a prefix of it
        ("conv2d", "/kernel"),
        ("batch_normalization", "/gamma"),
    ):
        hits = [
            c for c in consts
            if c.endswith(suffix) and c.split("/")[0].rstrip("_0123456789") == family
        ]
        idx = sorted(index_of(h) for h in hits)
        print(f"  {family:<20} n={len(idx):<3} indices={idx}")

    # The tensor feeding the U-Net, and the one coming out. Both are named by
    # position in the *whole* graph, so neither name is safe to hardcode.
    slices = [op.name for op in g.get_operations() if op.type == "StridedSlice"]
    muls = [op.name for op in g.get_operations() if op.name.endswith("_spectrogram/mul")]
    print(f"  StridedSlice ops: {slices}")
    print(f"  output mul ops:   {muls}")


if __name__ == "__main__":
    print(f"--- {sys.argv[1]}")
    main(sys.argv[1])
