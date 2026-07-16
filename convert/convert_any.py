#!/usr/bin/env python3
"""Port any Spleeter stem's U-Net from a frozen TF graph to PyTorch.

sherpa-onnx's convert_to_torch.py only reaches 2 stems, because it hardcodes TF
op-name offsets by hand (vocals -> conv2d/*, accompaniment -> conv2d_7/*,
bn_offset=12). Rather than extend the hardcoding to 4, this derives the mapping
from the graph: each frozen graph contains exactly one stem's U-Net, so collect
its Const ops, sort by index, and assign in order.

That works because of an observed fact (see inspect_graph.py): within a stem's
block TF allocates 12 BatchNorm indices but only 11 exist -- conv5 has none --
and *freezing prunes the unused one*. So the sorted list is exactly 11 long and
lines up with the U-Net's 11 BatchNorms. Verified to reproduce sherpa's
hand-written 2stems mapping.

Nothing here is trusted on faith: main() runs the TF graph and the ported torch
model on the same input and asserts they agree.
"""

import argparse
import re

import numpy as np
import tensorflow as tf
import torch

from unet_any import UNet

# Sorted-by-index order maps onto these, in this order.
ACTS = {"2stems": ("LeakyReLU", "ReLU"), "4stems": ("ELU", "ELU")}

CONVS = ["conv", "conv1", "conv2", "conv3", "conv4", "conv5", "up7"]      # 7
UPS = ["up1", "up2", "up3", "up4", "up5", "up6"]                          # 6
BNS = ["bn", "bn1", "bn2", "bn3", "bn4",
       "bn5", "bn6", "bn7", "bn8", "bn9", "bn10"]                         # 11


def index_of(name: str) -> int:
    """`conv2d/kernel` -> 0, `conv2d_transpose_6/kernel` -> 6."""
    head = name.split("/")[0]
    m = re.search(r"_(\d+)$", head)
    return int(m.group(1)) if m else 0


def family_of(name: str) -> str:
    """`conv2d_transpose_6/kernel` -> `conv2d_transpose`."""
    return name.split("/")[0].rstrip("_0123456789")


def load_graph(path):
    with tf.compat.v1.gfile.GFile(path, "rb") as f:
        gd = tf.compat.v1.GraphDef()
        gd.ParseFromString(f.read())
    with tf.Graph().as_default() as g:
        tf.import_graph_def(gd, name="")
    return g


def scopes(graph, family: str, suffix: str) -> list[str]:
    """Every op scope of `family`, in index order. `conv2d` never returns a
    `conv2d_transpose` -- family_of strips the digits, so the two are distinct
    strings rather than one being a prefix of the other."""
    names = [
        op.name for op in graph.get_operations()
        if op.type == "Const" and op.name.endswith(suffix)
        and family_of(op.name) == family
    ]
    return [n.split("/")[0] for n in sorted(names, key=index_of)]


def consts(graph, names: list[str]) -> dict[str, torch.Tensor]:
    with tf.compat.v1.Session(graph=graph) as sess:
        wanted = set(names)
        out = {}
        for op in sess.graph.get_operations():
            if op.type == "Const" and op.name in wanted:
                out[op.name] = torch.from_numpy(sess.run(op.outputs[0]))
        return out


def waveform():
    np.random.seed(20230821)
    return np.random.rand(60 * 44100).astype(np.float32).reshape(-1, 2)


@torch.no_grad()
def main(stem: str, model_dir: str) -> None:
    graph = load_graph(f"{model_dir}/frozen_{stem}_model.pb")

    conv_scopes = scopes(graph, "conv2d", "/kernel")
    up_scopes = scopes(graph, "conv2d_transpose", "/kernel")
    bn_scopes = scopes(graph, "batch_normalization", "/gamma")
    assert len(conv_scopes) == 7, conv_scopes
    assert len(up_scopes) == 6, up_scopes
    assert len(bn_scopes) == 11, bn_scopes
    print(f"  {stem:<8} conv={conv_scopes[0]}..{conv_scopes[-1]}  "
          f"up={up_scopes[0]}..{up_scopes[-1]}  bn={bn_scopes[0]}..{bn_scopes[-1]}")

    want = []
    for s in conv_scopes + up_scopes:
        want += [f"{s}/kernel", f"{s}/bias"]
    for s in bn_scopes:
        want += [f"{s}/{p}" for p in
                 ("gamma", "beta", "moving_mean", "moving_variance")]
    param = consts(graph, want)

    unet = UNet(*ACTS[model_dir.strip("./")])
    unet.eval()
    sd = unet.state_dict()

    # TF conv kernels are (kh, kw, in, out); torch wants (out, in, kh, kw).
    # TF transpose kernels are (kh, kw, out, in) and torch ConvTranspose2d wants
    # (in, out, kh, kw) -- so the same permute serves both.
    for key, scope in zip(CONVS + UPS, conv_scopes + up_scopes):
        sd[f"{key}.weight"] = param[f"{scope}/kernel"].permute(3, 2, 0, 1)
        sd[f"{key}.bias"] = param[f"{scope}/bias"]
    for key, scope in zip(BNS, bn_scopes):
        sd[f"{key}.weight"] = param[f"{scope}/gamma"]
        sd[f"{key}.bias"] = param[f"{scope}/beta"]
        sd[f"{key}.running_mean"] = param[f"{scope}/moving_mean"]
        sd[f"{key}.running_var"] = param[f"{scope}/moving_variance"]
    unet.load_state_dict(sd)

    # The check that makes all of the above safe: same input through TF and
    # through the port. strided_slice_3 is the spectrogram entering the U-Net;
    # it survives freezing under that name in every stem's graph.
    x = graph.get_tensor_by_name("waveform:0")
    y0 = graph.get_tensor_by_name("strided_slice_3:0")
    y1 = graph.get_tensor_by_name(f"{stem}_spectrogram/mul:0")
    with tf.compat.v1.Session(graph=graph) as sess:
        y0_out, y1_out = sess.run([y0, y1], feed_dict={x: waveform()})

    got = unet(torch.from_numpy(y0_out).permute(3, 0, 1, 2)).permute(1, 0, 2, 3)
    want_t = torch.from_numpy(y1_out).permute(0, 3, 1, 2)
    err = (got - want_t).abs().max().item()
    assert torch.allclose(got, want_t, atol=1e-1), f"{stem}: max err {err}"
    print(f"  {stem:<8} TF vs torch: max err {err:.2e}  OK")

    torch.save(unet.state_dict(), f"{model_dir}/{stem}.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model-dir", default="./4stems")
    a = p.parse_args()
    main(a.name, a.model_dir)
