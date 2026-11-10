#!/usr/bin/env python3
"""Annotate tinyx3_ivfs.onnx with the quantisation policy validated in requant_ab.py.

Three changes over attach_quant.py (per-tensor / absmax / all-int8):

  1. per-channel activation scales  (dims [1], attribute `*_scales` + `*_quant_dim`)
  2. p99.9 clipping instead of absmax
  3. int16 for the depthwise blocks

The decoder conv, the softmax chain and the network output stay scalar and
int8: a per-class scale on the logits would change the argmax itself.

    python attach_quant_v2.py --out ../../onnx_models/tinyx3_ivfs_q8.onnx

Scale convention matches the compiler reference file ctc_test.tinyx3.onnx:
q = round(real * scale), i.e. scale is a MULTIPLIER (scale = qmax / ref).
"""
import argparse
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

SCALE_ATTRS = ("input_scale", "output_scale", "exp_scale", "weight_ch_scales",
               "input_scales", "output_scales", "input_quant_dim",
               "output_quant_dim", "input_bitdepth", "output_bitdepth",
               "weight_bitdepth", "py_module_name", "py_module_type")


def stripped(model):
    m = onnx.ModelProto()
    m.CopyFrom(model)
    for n in m.graph.node:
        keep = [a for a in n.attribute if a.name not in SCALE_ATTRS]
        del n.attribute[:]
        n.attribute.extend(keep)
    return m


def calibrate(model, vecdir, pct):
    m = stripped(model)
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.extend([onnx.ValueInfoProto(name=o)])
                have.add(o)
    p = "/tmp/_calib_v2.onnx"
    onnx.save(m, p)
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]

    files = sorted(glob.glob(os.path.join(vecdir, "input_*.npy")))
    acc = {n: [] for n in names}
    acc["input"] = []
    for f in files:
        x = np.load(f)
        acc["input"].append(np.asarray(x).reshape(-1))
        for n, o in zip(names, sess.run(None, {"input": x})):
            acc[n].append(o)
    print("calibrated on %d vectors" % len(files))

    stats = {}
    for n in list(acc.keys()):
        L = acc.pop(n)
        if np.asarray(L[0]).ndim == 4:
            A = np.concatenate([a.reshape(a.shape[0], a.shape[1], -1) for a in L], axis=2)
        else:
            A = np.concatenate([np.asarray(a).reshape(1, 1, -1) for a in L], axis=2)
        ab = np.abs(A)
        stats[n] = dict(ch_amax=ab.max(axis=(0, 2)),
                        ch_pct=np.percentile(ab, pct, axis=(0, 2)),
                        amax=float(ab.max()),
                        pct=float(np.percentile(ab, pct)),
                        ndim=np.asarray(L[0]).ndim)
        del L, A, ab
    return stats


def bits_of_tensors(nodes):
    """8/16 per tensor: depthwise convs and their Relu consumers run in int16."""
    tb = {"input": 8}
    for n in nodes:
        if n.op_type == "Constant":
            continue
        in16 = tb.get(n.input[0], 8) == 16
        out16 = ("depthwise" in n.name) or (in16 and n.op_type == "Relu")
        tb[n.output[0]] = 16 if out16 else 8
    return tb


def add(node, name, value):
    for a in list(node.attribute):
        if a.name == name:
            node.attribute.remove(a)
    node.attribute.append(onnx.helper.make_attribute(name, value))


def set_quant(prefix, node, ref, bits, per_channel, qmax):
    add(node, "%s_bitdepth" % prefix, int(bits))
    amp = np.maximum(np.asarray(ref, np.float64), 1e-12)
    if per_channel:
        add(node, "%s_scales" % prefix, [float(x) for x in qmax / amp])
        add(node, "%s_quant_dim" % prefix, [1])
    else:
        add(node, "%s_scale" % prefix, float(qmax / float(amp.reshape(-1)[0])))


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--model", default=os.path.join(root, "onnx_models", "tinyx3_ivfs.onnx"))
    ap.add_argument("--vectors", default=os.path.join(root, "test_vectors", "tinyx3_ivfs_folded"))
    ap.add_argument("--out", default=os.path.join(root, "onnx_models", "tinyx3_ivfs_q8.onnx"))
    ap.add_argument("--pct", type=float, default=99.9)
    a = ap.parse_args()

    model = stripped(onnx.load(a.model))   # drop the previous annotation entirely
    stats = calibrate(model, a.vectors, a.pct)
    nodes = [n for n in model.graph.node]
    tb = bits_of_tensors(nodes)

    for n in nodes:
        if n.op_type == "Constant":
            continue
        bits_in = tb.get(n.input[0], 8)
        bits_out = tb[n.output[0]]

        def per_ch(t):
            st = stats[t]
            return st["ndim"] == 4 and np.asarray(st["ch_pct"]).size > 1 \
                and "decoder" not in n.name

        if n.op_type == "Conv":
            pc = per_ch(n.input[0])
            set_quant("input", n, stats[n.input[0]]["ch_pct" if pc else "pct"],
                      bits_in, pc, 127.0 if bits_in == 8 else 32767.0)
            wbits = 16 if "depthwise" in n.name else 8
            add(n, "weight_bitdepth", wbits)
            w = numpy_helper.to_array(
                [i for i in model.graph.initializer if i.name == n.input[1]][0])
            wm = np.abs(w.astype(np.float64)).reshape(w.shape[0], -1).max(axis=1)
            wq = 32767.0 if wbits == 16 else 127.0
            add(n, "weight_ch_scales", [float(x) for x in wq / np.maximum(wm, 1e-12)])
        elif n.op_type == "Softmax":
            sin = stats[n.input[0]]["pct"]
            sout = stats[n.output[0]]["amax"]
            set_quant("input", n, sin, 8, False, 127.0)
            set_quant("output", n, sout, 8, False, 127.0)
            add(n, "exp_scale", float((127.0 / sout) / (127.0 / sin)))
            add(n, "mode", 0)
            continue

        pco = per_ch(n.output[0])
        set_quant("output", n, stats[n.output[0]]["ch_pct" if pco else "pct"],
                  bits_out, pco, 127.0 if bits_out == 8 else 32767.0)

    onnx.save(model, a.out)
    print("wrote %s  (%d nodes)" % (a.out, len(model.graph.node)))

    # weights must be untouched
    src = {i.name: i.SerializeToString() for i in onnx.load(a.model).graph.initializer}
    dst = {i.name: i.SerializeToString() for i in model.graph.initializer}
    print("initializers unchanged:", src == dst)
    b16 = sum(1 for v in tb.values() if v == 16)
    print("int16 tensors: %d / %d" % (b16, len(tb)))


if __name__ == "__main__":
    main()

