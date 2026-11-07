#!/usr/bin/env python3
"""Export the trained cascade TINYX3 to a static ONNX graph with BN folded."""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from model_cascade import Model  # noqa: E402


def fuse_pair(conv, bn):
    fused = copy.deepcopy(conv)
    scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    b = -bn.running_mean * scale + bn.bias.data
    if conv.bias is not None:
        b = b + conv.bias.data * scale
    fused.weight.data.copy_(w)
    fused.bias = nn.Parameter(b.clone())
    return fused


def fuse_bn(model):
    pairs = [("conv0", "bn0")]
    pairs += [("pointwise%d" % i, "bn%d" % i) for i in range(1, 28)]
    pairs.append(("conv28", "bn28"))
    n = 0
    for conv_name, bn_name in pairs:
        conv = getattr(model, conv_name)
        bn = getattr(model, bn_name)
        setattr(model, conv_name, fuse_pair(conv, bn))
        setattr(model, bn_name, nn.Identity())
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(REPO, "checkpoints", "tinyx3_ivfs.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "onnx", "TINYX3_cascade.onnx"))
    ap.add_argument("--T", type=int, default=1334)
    ap.add_argument("--golden", default="")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model = Model()
    model.load_state_dict(sd)
    model.eval()
    n_fused = fuse_bn(model)
    print("[fuse] conv+bn pairs:", n_fused)

    dummy = torch.randn(1, 1, 3, args.T)
    with torch.no_grad():
        ref = model(dummy)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["prob"],
        opset_version=13, do_constant_folding=True,
    )
    print("[export]", args.out, "input", tuple(dummy.shape), "output", tuple(ref.shape))

    import onnx
    import onnxruntime as ort
    from collections import Counter

    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    ops = Counter(n.op_type for n in m.graph.node)
    print("[onnx] ops:", dict(sorted(ops.items())))

    kernels = []
    for node in m.graph.node:
        if node.op_type == "Conv":
            for a in node.attribute:
                if a.name == "kernel_shape":
                    kernels.append(int(a.ints[-1]))
    print("[onnx] conv count:", len(kernels), "max kernel W:", max(kernels),
          "kernel hist:", dict(sorted(Counter(kernels).items())))

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": dummy.numpy()})[0]
    md = float(np.max(np.abs(out - ref.numpy())))
    print("[verify] max|onnx-torch| = %.3e" % md)

    if args.golden:
        xin = torch.from_numpy(np.load(args.golden.replace("output_", "input_")))
        yout = torch.from_numpy(np.load(args.golden))
        with torch.no_grad():
            ytorch = model(xin).numpy()
        og = sess.run(None, {"input": xin.numpy()})[0]
        print("[golden] in", tuple(xin.shape), "out", tuple(yout.shape))
        print("[golden] max|torch-golden| = %.3e" % float(np.max(np.abs(ytorch - yout.numpy()))))
        print("[golden] max|onnx-golden|  = %.3e" % float(np.max(np.abs(og - yout.numpy()))))


if __name__ == "__main__":
    main()
