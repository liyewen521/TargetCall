#!/usr/bin/env python3
"""Read an annotated ONNX back and emulate its int8/int16 pipeline.

Verifies that the file produced by attach_quant_v2.py is self-consistent and
measures accuracy against the float golden outputs.

    python check_quant.py --model ../../onnx_models/tinyx3_ivfs_q8.onnx
"""
import argparse
import os

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper


def load(path):
    m = onnx.load(path)
    ini = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        d = {}
        for a in n.attribute:
            if a.name in ("weight_ch_scales",) or a.name.endswith("_scales"):
                d[a.name] = np.asarray(list(a.floats), np.float64)
            elif a.name.endswith("_scale"):
                d[a.name] = float(a.f)
            elif a.name.endswith("_bitdepth"):
                d[a.name] = int(a.i)
            elif len(a.ints):
                d[a.name] = list(a.ints)
            elif a.type == onnx.AttributeProto.INT:
                d[a.name] = [a.i]
        nodes.append((n, d))
    return nodes, ini


def bc(s, ndim):
    s = np.asarray(s, np.float64)
    if s.size == 1:
        return float(s.reshape(-1)[0])
    shape = [1] * ndim
    shape[1] = s.size
    return s.reshape(shape)


def emulate(nodes, ini, x):
    val = {"input": x}
    for n, d in nodes:
        op = n.op_type
        if op == "Constant":
            for a in n.attribute:
                if a.name in ("t", "value"):
                    val[n.output[0]] = numpy_helper.to_array(a.t)
            continue
        if op == "Conv":
            s_in = d.get("input_scales", d.get("input_scale"))
            s_out = d.get("output_scales", d.get("output_scale"))
            qmax = (1 << d["output_bitdepth"]) - 1
            s_w = d["weight_ch_scales"]
            w = ini[n.input[1]].astype(np.float64)
            s4 = s_w.reshape(-1, 1, 1, 1)
            wd = np.rint(w * s4) / s4
            pads = d.get("pads", [0, 0, 0, 0])
            st = d.get("strides", [1, 1])
            xin = val[n.input[0]].astype(np.float64) / bc(s_in, 4)
            y = F.conv2d(torch.from_numpy(xin), torch.from_numpy(wd), None,
                         padding=(int(pads[0]), int(pads[1])),
                         stride=(int(st[0]), int(st[1])),
                         groups=int(d.get("group", [1])[0])).numpy()
            if len(n.input) > 2:
                y = y + ini[n.input[2]].astype(np.float64).reshape(1, -1, 1, 1)
            val[n.output[0]] = np.clip(np.rint(y * bc(s_out, 4)), -qmax - 1, qmax)
        elif op == "Relu":
            s_in = d_in(nodes, n.input[0])
            s_out = d.get("output_scales", d.get("output_scale"))
            qmax = (1 << d["output_bitdepth"]) - 1
            y = np.maximum(val[n.input[0]].astype(np.float64) / bc(s_in, 4), 0.0)
            val[n.output[0]] = np.clip(np.rint(y * bc(s_out, 4)), -qmax - 1, qmax)
        elif op == "Reshape":
            v = val[n.input[0]]
            shape = np.asarray(val[n.input[1]]).astype(np.int64).tolist()
            shape = [v.shape[i] if s == 0 else s for i, s in enumerate(shape)]
            val[n.output[0]] = v.reshape(shape)
        elif op == "Transpose":
            perm = d.get("perm", list(range(val[n.input[0]].ndim)))
            val[n.output[0]] = np.transpose(val[n.input[0]], perm)
        elif op == "Softmax":
            s_in = d["input_scale"]
            s_out = d["output_scale"]
            qmax = (1 << d["output_bitdepth"]) - 1
            z = val[n.input[0]].astype(np.float64) / bc(s_in, 4)
            p = torch.softmax(torch.from_numpy(z), dim=-1).numpy()
            val[n.output[0]] = np.clip(np.rint(p * bc(s_out, 4)), -qmax - 1, qmax)
        else:
            raise SystemExit("unhandled " + op)
    return val[nodes[-1][0].output[0]]


_OUT = {}


def d_in(nodes, tensor):
    """Scale of the tensor, taken from whoever produced it."""
    if tensor in _OUT:
        return _OUT[tensor]
    for n, d in nodes:
        if n.output and n.output[0] == tensor:
            s = d.get("output_scales", d.get("output_scale"))
            _OUT[tensor] = s
            return s
    return 1.0


def decode(q):
    a = np.argmax(np.asarray(q).reshape(-1, 5), 1)
    out = [a[0]]
    for v in a[1:]:
        if v != out[-1]:
            out.append(v)
    return [v for v in out if v != 0]


def edit_identity(s1, s2):
    n, m = len(s1), len(s2)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (s1[i - 1] != s2[j - 1]))
        prev = cur
    return 1.0 - prev[m] / max(n, m)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--model", default=os.path.join(root, "onnx_models", "tinyx3_ivfs_q8.onnx"))
    ap.add_argument("--vectors", default=os.path.join(root, "test_vectors", "tinyx3_ivfs_folded"))
    ap.add_argument("--from", dest="lo", type=int, default=40)
    ap.add_argument("--to", dest="hi", type=int, default=49)
    a = ap.parse_args()

    nodes, ini = load(a.model)
    s0 = nodes[0][1].get("input_scales", nodes[0][1].get("input_scale"))
    idt, agr = [], []
    for i in range(a.lo, a.hi + 1):
        x = np.load(os.path.join(a.vectors, "input_%03d.npy" % i)).astype(np.float64)
        x = np.clip(np.rint(x * bc(s0, 4)), -128, 127)
        out = emulate(nodes, ini, x)
        f = np.load(os.path.join(a.vectors, "output_%03d.npy" % i))
        agr.append(np.mean(np.argmax(out.reshape(-1, 5), 1) == np.argmax(f.reshape(-1, 5), 1)))
        idt.append(edit_identity(decode(out), decode(f)))
        print("  vector %03d: argmax %.2f%%  identity %.2f%%"
              % (i, 100 * agr[-1], 100 * idt[-1]))
    print("mean over %d vectors: argmax %.2f%%  identity %.2f%%"
          % (len(idt), 100 * np.mean(agr), 100 * np.mean(idt)))


if __name__ == "__main__":
    main()

