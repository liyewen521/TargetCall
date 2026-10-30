#!/usr/bin/env python3
"""Hardware-friendly TINYX3 front-end: replace the stride-3 conv0 with a
stride-1 3x3 conv over a folded input.

Original conv0
    Conv2d(1, 12, kernel=(1,9), stride=(1,3), padding=(0,4))
Folded equivalent (numerically identical)
    input   raw signal [1,1,1,4000]  ->  fold  ->  [1,1,3,1334]   (ifmap 1x3x1334)
    conv0   Conv2d(1, 12, kernel=(3,3), stride=(1,1), padding=(0,1))

Fold (with zero padding that reproduces the original pad=4):
    y = [0] + x + [0]          # length 4002 = 3 * 1334
    M[r, c] = y[3*c + r]       # interleaved, row r = residue r  ->  shape (3, 1334)

Weight reorder (9 taps -> 3x3):
    W[oc, 0, r, k] = w[oc, 0, 0, 3*k + r]
    == w.reshape(12, 1, 3, 3).permute(0, 1, 3, 2)
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from inference import create_model
from convert_legacy import flatten_legacy_state_dict

MODEL = "TINYX3"
L = 4000
T = 1334

SRC_VEC = os.path.join(HERE, "test_vectors", "TINYX3")
OUT_VEC = os.path.join(HERE, "test_vectors", "TINYX3_folded")
ONNX_OUT = os.path.join(HERE, "onnx_models", "TINYX3_folded.onnx")


def fold_input(x):
    """raw [N,1,1,L] -> folded [N,1,3,T]  (M[r,c] = y[3c+r], y=[0]+x+[0])."""
    n = x.shape[0]
    z = torch.zeros(n, 1, 1, 1, dtype=x.dtype)
    y = torch.cat([z, x, z], dim=3)               # N,1,1,4002
    return y.reshape(n, 1, T, 3).transpose(2, 3)  # N,1,3,1334


def fold_conv0_weight(w):
    """(12,1,1,9) -> (12,1,3,3) with W[oc,0,r,k] = w[oc,0,0,3k+r]."""
    return w.reshape(12, 1, 3, 3).permute(0, 1, 3, 2).contiguous()


def build_model():
    sd = flatten_legacy_state_dict(
        MODEL, torch.load(os.path.join(HERE, "checkpoints", MODEL, "weights_1.tar"), map_location="cpu")
    )
    model = create_model(MODEL)
    model.load_state_dict(sd, strict=True)
    conv0 = nn.Conv2d(1, 12, (3, 3), stride=(1, 1), padding=(0, 1), bias=False)
    conv0.weight.data = fold_conv0_weight(sd["conv0.weight"])
    model.conv0 = conv0
    model.eval()
    return model


def main():
    model = build_model()

    # 1) verify against the 50 stored golden vectors
    ins = sorted(glob.glob(os.path.join(SRC_VEC, "input_*.npy")))
    outs = sorted(glob.glob(os.path.join(SRC_VEC, "output_*.npy")))
    worst = 0.0
    tot = mism = 0
    for ip, op in zip(ins, outs):
        x = torch.from_numpy(np.load(ip)).float()
        ref = np.load(op)
        with torch.no_grad():
            y = model(fold_input(x)).numpy()
        worst = max(worst, float(np.abs(y - ref).max()))
        tot += ref[..., 0].size
        mism += int((y.argmax(-1) != ref.argmax(-1)).sum())
    print("verify %d cases: worst prob diff %.3e | argmax %d/%d mismatch"
          % (len(ins), worst, mism, tot))

    # 2) write folded golden vectors (inputs folded, outputs = original reference)
    os.makedirs(OUT_VEC, exist_ok=True)
    for p in glob.glob(os.path.join(OUT_VEC, "*.npy")):
        os.remove(p)
    lines = ["idx\tread_id\tchunk"]
    cases = open(os.path.join(SRC_VEC, "cases.tsv")).read().splitlines()
    for i, (ip, op) in enumerate(zip(ins, outs)):
        x = torch.from_numpy(np.load(ip)).float()
        fin = fold_input(x)
        np.save(os.path.join(OUT_VEC, "input_%03d.npy" % i), fin.numpy().astype("<f4"))
        np.save(os.path.join(OUT_VEC, "output_%03d.npy" % i), np.load(op).astype("<f4"))
        if i + 1 < len(cases):
            lines.append(cases[i + 1])
    with open(os.path.join(OUT_VEC, "cases.tsv"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote folded vectors ->", OUT_VEC, "input", tuple(fin.shape))

    # 3) export ONNX with folded input layout [1,1,3,1334]
    os.makedirs(os.path.dirname(ONNX_OUT), exist_ok=True)
    dummy = torch.randn(1, 1, 3, T)
    torch.onnx.export(
        model, dummy, ONNX_OUT,
        input_names=["signal"], output_names=["log_probs"],
        dynamic_axes={"signal": {0: "batch", 3: "length"}, "log_probs": {0: "batch", 2: "time"}},
        opset_version=14, do_constant_folding=True,
    )
    print("exported onnx ->", ONNX_OUT)


if __name__ == "__main__":
    main()
