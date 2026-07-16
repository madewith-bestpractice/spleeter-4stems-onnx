#!/usr/bin/env python3
"""Where do TF and the torch port diverge?

The port is not in doubt as *code*: convert_any.py reproduces sherpa's mapping
and passes on 2stems (3e-4). The same code on 4stems is off by 945, and the two
graphs have identical architecture and identical op-name layout. So compare the
intermediates and find the first layer that disagrees.
"""

import sys

import numpy as np
import tensorflow as tf
import torch

from convert_any import load_graph, scopes, waveform
from unet import UNet


def tf_run(graph, names, feed):
    with tf.compat.v1.Session(graph=graph) as sess:
        x = graph.get_tensor_by_name("waveform:0")
        return sess.run([graph.get_tensor_by_name(n) for n in names],
                        feed_dict={x: feed})


def main(tag: str) -> None:
    graph = load_graph(f"{tag}/frozen_vocals_model.pb")
    unet = UNet()
    unet.load_state_dict(torch.load(f"{tag}/vocals.pt", map_location="cpu"))
    unet.eval()

    conv_s = scopes(graph, "conv2d", "/kernel")

    # Probe points: input, first conv pre-activation, first BN, first leaky relu.
    names = [
        "strided_slice_3:0",
        f"{conv_s[0]}/BiasAdd:0",
    ]
    # The BN op name varies (cond/FusedBatchNorm vs FusedBatchNormV3); find it.
    bn_candidates = [
        op.name for op in graph.get_operations()
        if op.name.startswith("batch_normalization/") and "FusedBatchNorm" in op.type
    ]
    print(f"[{tag}] BN op type(s): "
          f"{[(o.name, o.type) for o in graph.get_operations() if o.name.startswith('batch_normalization/') and 'Fused' in o.type]}")
    if bn_candidates:
        names.append(f"{bn_candidates[0]}:0")

    outs = tf_run(graph, names, waveform())
    spec = torch.from_numpy(outs[0]).permute(3, 0, 1, 2)   # (ch, splits, T, F)

    # torch: replicate UNet.forward up to the first conv + bn
    x = spec.permute(1, 0, 2, 3)                            # (splits, ch, T, F)
    xp = torch.nn.functional.pad(x, (1, 2, 1, 2), "constant", 0)
    with torch.no_grad():
        t_conv = unet.conv(xp)
        t_bn = unet.bn(t_conv)

    tf_conv = torch.from_numpy(outs[1]).permute(0, 3, 1, 2)
    print(f"[{tag}] conv0  TF {tuple(tf_conv.shape)} torch {tuple(t_conv.shape)}  "
          f"max err {(tf_conv - t_conv).abs().max():.3e}")
    if len(outs) > 2:
        tf_bn = torch.from_numpy(outs[2]).permute(0, 3, 1, 2)
        print(f"[{tag}] bn0    max err {(tf_bn - t_bn).abs().max():.3e}")

    # And the BN's own constants -- the usual suspect when conv agrees and bn does not.
    with tf.compat.v1.Session(graph=graph) as sess:
        for p in ("gamma", "beta", "moving_mean", "moving_variance"):
            v = sess.run(graph.get_tensor_by_name(f"batch_normalization/{p}:0"))
            print(f"[{tag}] bn0 {p:<17} mean {v.mean():+.4f}  std {v.std():.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
